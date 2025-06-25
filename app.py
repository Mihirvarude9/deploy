from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from starlette.concurrency import run_in_threadpool
from uuid import uuid4
import asyncio
import torch
import os
from pathlib import Path

# ───────────────────── CONFIG ─────────────────────
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUTPUT_DIR = Path("generated")
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_PARALLEL = 3
NUM_STEPS = 50
GUIDANCE = 5.5

# ────────────── FASTAPI SETUP ──────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")

# ────────────── MODEL LOAD ──────────────
print("🔄 Loading Stable Diffusion 3.5 Medium...")
transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_model_cpu_offload()
print("✅ Model loaded and ready.")

# ────────────── CONCURRENCY SETUP ──────────────
semaphore = asyncio.Semaphore(MAX_PARALLEL)

# ────────────── SCHEMAS ──────────────
class PromptRequest(BaseModel):
    prompt: str

# ────────────── HELPER FUNCTIONS ──────────────
def run_generation(prompt: str) -> str:
    image = pipe(
        prompt=prompt,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE
    ).images[0]
    
    filename = f"{uuid4().hex}.png"
    filepath = OUTPUT_DIR / filename
    image.save(filepath)
    return filename

async def generate_image(prompt: str) -> str:
    async with semaphore:
        filename = await run_in_threadpool(run_generation, prompt)
        return filename

# ────────────── ROUTES ──────────────
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/generate/", response_model=dict)
async def generate(request: Request, body: PromptRequest):
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt is empty")

    try:
        filename = await generate_image(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return JSONResponse(
        {"image_url": f"https://api.wildmindai.com/images/{filename}"}
    )
