# sd_multigpu.py  ───────────────────────────────────────────────────────────────
import os, itertools, asyncio
from functools    import partial
from uuid         import uuid4
from typing       import Dict

import torch
from fastapi             import FastAPI, Request, HTTPException
from pydantic            import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses   import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from diffusers           import (
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

MODEL_ID   = "stabilityai/stable-diffusion-3.5-medium"
API_KEY    = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUT_DIR    = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(OUT_DIR, exist_ok=True)

################################################################################
# 1.  GPU-level bookkeeping
################################################################################
GPU_IDS          = list(range(torch.cuda.device_count()))
gpu_active_jobs  = {gid: 0 for gid in GPU_IDS}         # how many requests run
gpu_lock         = asyncio.Lock()                      # guards gpu_active_jobs


def _choose_gpu_round_robin() -> int:
    """
    Return the GPU id that currently has *least* active work.
    Called inside gpu_lock → thread-safe.
    """
    return min(gpu_active_jobs, key=gpu_active_jobs.get)


async def acquire_gpu() -> int:
    """
    Atomically pick a GPU and mark one job active.
    """
    async with gpu_lock:
        gid = _choose_gpu_round_robin()
        gpu_active_jobs[gid] += 1
        return gid


def release_gpu(gid: int) -> None:
    """Decrement active-job counter (runs in thread, no async needed)."""
    gpu_active_jobs[gid] -= 1


################################################################################
# 2.  One pipeline object *per GPU*  (lazy-loaded on first use)
################################################################################
PIPELINES: Dict[int, StableDiffusion3Pipeline] = {}
PIPELINE_LOCK = asyncio.Lock()   # ensures we create a pipeline only once


def _get_pipeline(gid: int) -> StableDiffusion3Pipeline:
    """
    Create & cache a pipeline *on that GPU* the first time it is requested.
    Subsequent calls return the cached instance.
    """
    if gid in PIPELINES:
        return PIPELINES[gid]

    with torch.cuda.device(gid):
        print(f"🔄  Loading SD-3.5-medium on GPU {gid} …")
        transformer = SD3Transformer2DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.float16
        ).to(f"cuda:{gid}")

        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.float16
        ).to(f"cuda:{gid}")

        pipe.enable_model_cpu_offload()   # VRAM saver
        PIPELINES[gid] = pipe
        print(f"✅  GPU {gid} ready!")
        return pipe


################################################################################
# 3.  One request  →  one scheduler copy  →  render in thread
################################################################################
def _render(gid: int, prompt: str) -> str:
    pipe = _get_pipeline(gid)

    # give this call its own scheduler
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.scheduler.set_timesteps(50, device=pipe.device)

    img = pipe(prompt, num_inference_steps=50,
               guidance_scale=5.5).images[0]

    fname = f"{uuid4().hex}.png"
    img.save(os.path.join(OUT_DIR, fname))
    return fname


async def generate_image(prompt: str) -> str:
    gid = await acquire_gpu()             # ↙ async – no blocking here
    try:
        filename = await run_in_threadpool(partial(_render, gid, prompt))
        return filename
    finally:
        release_gpu(gid)


################################################################################
# 4.  FastAPI glue
################################################################################
class Prompt(BaseModel):
    prompt: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=OUT_DIR), name="images")


@app.post("/generate")
async def generate(req: Request, body: Prompt):
    if req.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    filename = await generate_image(prompt)
    return JSONResponse(
        {"image_url": f"https://api.wildmindai.com/images/{filename}"}
    )


@app.get("/healthz")        # simple liveness probe
def health():
    return {"status": "ok", "gpu_jobs": gpu_active_jobs}
