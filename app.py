# app_async.py  ───────────────────────────────────────────────────────────
import os, uuid, threading, concurrent.futures, traceback

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline

# ─────────────── CONFIG ─────────────────────────────────────────────────
API_KEY   = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
MODEL_ID  = "stabilityai/stable-diffusion-3.5-medium"
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORKERS = 4          # <- tweak to match your GPU(s) / CPU cores
GUIDANCE    = 5.5
STEPS       = 50

# ─────────────── LOAD MODEL once at start-up ────────────────────────────
print("🔄 Loading SD-3.5-medium …")
transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID, transformer=transformer, torch_dtype=torch.float16
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print("✅ Model ready")

# ─────────────── FASTAPI BOILERPLATE ────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# ─────────────── IN-MEMORY JOB REGISTRY ────────────────────────────────
JOBS = {}                # job_id → dict(status, filename or error)
LOCK = threading.Lock()  # protect JOBS

executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

def _render(job_id: str, prompt: str) -> None:
    """Runs in a worker thread."""
    try:
        with LOCK:
            JOBS[job_id]["status"] = "processing"

        img = pipe(prompt, num_inference_steps=STEPS,
                   guidance_scale=GUIDANCE).images[0]

        filename = f"{job_id}.png"
        img.save(os.path.join(OUTPUT_DIR, filename))

        with LOCK:
            JOBS[job_id].update(status="done", filename=filename)

    except Exception as e:              # log & surface error
        traceback.print_exc()
        with LOCK:
            JOBS[job_id].update(status="error", error=str(e))

# ─────────────── REQUEST SCHEMA ────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str

def _check_key(req: Request):
    if req.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ─────────────── ROUTES ────────────────────────────────────────────────
@app.post("/generate")
async def generate(req: Request, body: PromptRequest):
    _check_key(req)
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    job_id = uuid.uuid4().hex
    with LOCK:
        JOBS[job_id] = {"status": "queued"}

    executor.submit(_render, job_id, prompt)

    return {"job_id": job_id,
            "status_url": f"/result/{job_id}"}

@app.get("/result/{job_id}")
async def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")

        if job["status"] == "done":
            return {"status": "done",
                    "image_url": f"https://api.wildmindai.com/images/{job['filename']}"}
        elif job["status"] == "error":
            return {"status": "error", "detail": job["error"]}
        else:
            return {"status": job["status"]}

# ───────────────── OPTIONAL compatibility endpoint ─────────────────────
@app.post("/generate_sync")
async def generate_sync(req: Request, body: PromptRequest):
    """Blocks; behaves like your original endpoint."""
    _check_key(req)
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    img = pipe(prompt, num_inference_steps=STEPS,
               guidance_scale=GUIDANCE).images[0]
    filename = f"{uuid.uuid4().hex}.png"
    img.save(os.path.join(OUTPUT_DIR, filename))

    return {"image_url": f"https://api.wildmindai.com/images/{filename}"}
