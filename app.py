from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-medium"

# Ensure dtype consistency
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # Changed to float16 to match expected dtype
)

model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16  # Use float16 instead of bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.float16  # Ensure consistency
)
pipeline.enable_model_cpu_offload()

# Shorten the prompt manually to fit within 77 CLIP tokens
prompt = "Create a photorealistic image of an international astronaut on Mars, wearing a giant duck inflatable toy around its waist"

# Generate the image
image = pipeline(
    prompt=prompt,
    num_inference_steps=60,
    guidance_scale=5.5,
).images[0]

image.save("whimsical.png")
