## updated 13/7/25  to create a final image (this helps if the original image is very blurry or highly pixelated)

import os
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_IMAGE      = "output_3.png"        # Place this file in your working dir
OUTPUT_IMAGE     = "output_4.png"
IP_ADAPTER_PATH  = "models/ip-adapter-plus-face_sd15.bin"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# How many times to repeat the img2img pass
NUM_ITERS        = 1

# Img2Img settings
INIT_RES         = (512, 512)
STRENGTH         = 0.6
GUIDANCE_SCALE   = 8.0
STEPS            = 148

# Prompts
PROMPT           = (
    "ultra-high-definition portrait, smooth skin texture, sharp facial features, eyes, nose, "
    "cinematic lighting, professional studio quality, 16k resolution, flawless complexion, "
    "detailed eyes, razor-sharp eyes, same age, same gender, perfect symmetry, no pixelation, anti-aliased, f/1.4 aperture"
)
NEG_PROMPT       = (
    "pixelated, jagged edges, grainy, noisy, blurry, cartoon, jpeg artifacts, anime, blocking, "
    "deformed, distorted, disfigured, bad anatomy, text, gender swap, age change, makeup, signature, watermark, low quality, artifacts"
)


# ─── SANITY CHECK ────────────────────────────────────────────────────────────────
if not os.path.exists(INPUT_IMAGE):
    raise FileNotFoundError(f"Place '{INPUT_IMAGE}' in your working directory and re-run.")

# ─── LOAD PIPELINE ──────────────────────────────────────────────────────────────
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    safety_checker=None,
).to(DEVICE)

# attach the IP-Adapter
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name=os.path.basename(IP_ADAPTER_PATH),
)
pipe.set_ip_adapter_scale(1.0)

# ─── PREPARE INITIAL IMAGE ─────────────────────────────────────────────────────
img = Image.open(INPUT_IMAGE).convert("RGB")
img = img.resize(INIT_RES, Image.LANCZOS)

# ─── ITERATIVE IMG2IMG ─────────────────────────────────────────────────────────
for i in range(1, NUM_ITERS+1):
    print(f"▶ Iteration {i}/{NUM_ITERS} ...", end=" ")
    with torch.autocast(DEVICE):
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=img,
            ip_adapter_image=img,
            strength=STRENGTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=STEPS,
            generator=torch.Generator(DEVICE).manual_seed(42 + i)
        ).images[0]
    img = out
    print("done")

# ─── SAVE FINAL OUTPUT ──────────────────────────────────────────────────────────
img.save(OUTPUT_IMAGE)
print(f"✅ Saved final → {OUTPUT_IMAGE}") 