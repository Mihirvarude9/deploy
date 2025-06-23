# === app.py (FastAPI with GPU concurrency support) ===
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
import os

# === CONFIG ===
model_id = "stabilityai/stable-diffusion-3.5-medium"
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === GPU Assignment ===
gpu_count = torch.cuda.device_count()
worker_id = int(os.getpid()) % gpu_count
print(f"[Worker PID: {os.getpid()}] Using GPU {worker_id}")
torch.cuda.set_device(worker_id)

# === LOAD MODEL ===
model = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.float16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model,
    torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# === FASTAPI SETUP ===
app = FastAPI()

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static images from /images
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# === Request Schema ===
class PromptRequest(BaseModel):
    prompt: str

# === /generate endpoint ===
@app.post("/generate")
async def generate(request: Request, body: PromptRequest, background_tasks: BackgroundTasks):
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    def run_generation():
        with torch.inference_mode():
            image = pipeline(prompt=prompt, num_inference_steps=50, guidance_scale=5.5).images[0]
            image.save(filepath)

    background_tasks.add_task(run_generation)

    return {"image_url": f"https://api.wildmindai.com/images/{filename}"}
