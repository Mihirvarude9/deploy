"""
Start one copy *per GPU*, e.g.

    CUDA_VISIBLE_DEVICES=0 uvicorn sd_app:app --host 127.0.0.1 --port 7900 &
    CUDA_VISIBLE_DEVICES=1 uvicorn sd_app:app --host 127.0.0.1 --port 7901 &
    …
"""

import os, asyncio, traceback
from functools import partial
from uuid import uuid4
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline

# ─────────────────────────────────── CONFIG
MODEL_ID: Literal["stabilityai/stable-diffusion-3.5-medium"] = \
          "stabilityai/stable-diffusion-3.5-medium"
API_KEY       = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUT_DIR       = os.path.abspath("generated")
MAX_PARALLEL_JOBS = 4        # ⬅︎  how many prompts may run simultaneously

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────── MODEL (one per GPU/process)
print("🔄  Loading SD-3.5-medium …")
_transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID, transformer=_transformer, torch_dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()          # small memory saving
pipe.to(memory_format=torch.channels_last)
print("✅  Model ready on", torch.cuda.get_device_name(0))

# ─────────────────────────────────── FASTAPI
app = FastAPI(title="SD-3.5-medium worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=OUT_DIR), name="images")

# -------- schema
class Prompt(BaseModel):
    prompt: str

# -------- helpers
_sema = asyncio.Semaphore(MAX_PARALLEL_JOBS)

def _render(prompt: str) -> str:
    """Executed inside a thread; may raise OOM."""
    img = pipe(prompt, num_inference_steps=50,
               guidance_scale=5.5).images[0]
    fname = f"{uuid4().hex}.png"
    img.save(os.path.join(OUT_DIR, fname))
    return fname

async def generate_safe(prompt: str) -> str:
    async with _sema:                      # limit in-GPU concurrency
        try:
            return await run_in_threadpool(partial(_render, prompt))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GPU busy – please retry.",
            ) from e
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {e}",
            ) from e

# -------- route
@app.post("/generate")
async def generate(req: Request, body: Prompt):
    if req.headers.get("x-api-key") != API_KEY:
        raise HTTPException(401, "Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Prompt is empty")

    filename = await generate_safe(prompt)
    url = f"https://api.wildmindai.com/images/{filename}"
    return JSONResponse({"image_url": url})

# tiny health check
@app.get("/ping")
def ping():
    return {"ok": True}
