from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from diffusers import FluxPipeline
from uuid import uuid4
import torch
import os

# === CONFIG ===
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FastAPI instance ===
app = FastAPI()

# === CORSMiddleware must be added before routes ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],  # Your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Serve images ===
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# === Load FLUX Model ===
print("🔄 Loading FLUX Dev pipeline...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅ FLUX Dev model ready!")

# === Input Schema ===
class PromptRequest(BaseModel):
    prompt: str
    height: int = 512
    width: int = 512
    steps: int = 50
    guidance: float = 6.5
    seed: int = 42

# === /flux endpoint ===
@app.post("/generate")
async def generate_flux_image(request: Request, data: PromptRequest):
    # API Key check
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not data.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")

    generator = torch.manual_seed(data.seed)
    image = pipe(
        prompt=data.prompt,
        height=data.height,
        width=data.width,
        guidance_scale=data.guidance,
        num_inference_steps=data.steps,
        generator=generator
    ).images[0]

    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return JSONResponse({"image_url": f"https://api.wildmindai.com/images/{filename}"})
