"""
Run one worker instance per GPU (example for GPU-0)

CUDA_VISIBLE_DEVICES=0 uvicorn app:app --host 0.0.0.0 --port 7861 --workers 1
"""
import os
import asyncio
from uuid import uuid4
from functools import partial
from typing import Literal

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline

# ────────────────────────────── CONFIG ──────────────────────────────
MODEL_ID:     Literal["stabilityai/stable-diffusion-3.5-medium"] = \
              "stabilityai/stable-diffusion-3.5-medium"
API_KEY  = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUT_DIR  = os.path.abspath("generated")
os.makedirs(OUT_DIR, exist_ok=True)

# How many thread-workers may run *simultaneously*
MAX_PARALLEL_JOBS = 4          # tune for your GPU RAM

# ────────────────────────────── GPU MODEL ───────────────────────────
print("🔄  Loading SD-3.5 medium…")
_transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID, transformer=_transformer, torch_dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()          # a little more memory-friendly
pipe.to(memory_format=torch.channels_last)

print("✅  Model ready!")

# Optional lock if you *really* must run only one job on the GPU
# MODEL_LOCK = asyncio.Lock()

# ────────────────────────────── FASTAPI ─────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=OUT_DIR), name="images")

class PromptBody(BaseModel):
    prompt: str

# ───────────────────────────  UTILITIES  ────────────────────────────
def _generate_image(prompt: str) -> str:
    """Heavy, blocking call executed inside a thread."""
    # If you need strict serialisation, uncomment the lock
    # async with MODEL_LOCK:
    image = pipe(prompt, num_inference_steps=50,
                 guidance_scale=5.5).images[0]
    fname = f"{uuid4().hex}.png"
    fpath = os.path.join(OUT_DIR, fname)
    image.save(fpath)
    return fname

# Thread-pool limited to N jobs
_pool = asyncio.Semaphore(MAX_PARALLEL_JOBS)

async def generate_job(prompt: str) -> str:
    async with _pool:                # limit parallelism
        fname = await run_in_threadpool(partial(_generate_image, prompt))
        return fname

# ─────────────────────────── ENDPOINTS ──────────────────────────────
@app.post("/generate")
async def generate(request: Request, body: PromptBody):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(401, "Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Prompt is empty")

    try:
        filename = await generate_job(prompt)
    except Exception as err:
        raise HTTPException(500, f"Generation failed: {err}") from err

    url = f"https://api.wildmindai.com/images/{filename}"
    return JSONResponse({"image_url": url})
