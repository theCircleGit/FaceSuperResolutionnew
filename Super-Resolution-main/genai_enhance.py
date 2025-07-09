import os
import torch
import itertools
import numpy as np
import gc
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import threading

# --- GLOBALS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IP_ADAPTER_PATH = "models/ip-adapter-plus-face_sd15.bin"

# Simple lock to prevent concurrent requests
_processing_lock = threading.Lock()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

# For IP-Adapter
ip_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)
if hasattr(ip_pipe, 'load_ip_adapter'):
    ip_pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name=os.path.basename(IP_ADAPTER_PATH),
    )
    ip_pipe.set_ip_adapter_scale(1.0)

# --- CONFIG ---
STRENGTHS        = [0.25, 0.35, 0.45]
GUIDANCE_SCALES  = [5.0, 7.0, 9.0]
LOWRES           = (320, 320)
HIGHRES          = (768, 768)
FACE_ENHANCE_STRENGTH = 0.7
PROMPT = (
    "ultra-high-definition portrait, smooth skin texture, sharp facial features, eyes, nose, "
    "cinematic lighting, professional studio quality, 16k resolution, flawless complexion, "
    "detailed eyes, razor-sharp eyes, perfect symmetry, same age, same gender, no pixelation, anti-aliased, f/1.4 aperture"
)
NEG_PROMPT = (
    "pixelated, jagged edges, grainy, noisy, blurry, cartoon, jpeg artifacts, anime, blocking, "
    "deformed, distorted, disfigured, bad anatomy, text, signature, gender swap, age change, makeup, watermark, low quality, artifacts"
)

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def detect_face(image):
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            return x, y, x + w, y + h
    except:
        pass
    w, h = image.size
    return w // 4, h // 4, w * 3 // 4, h * 3 // 4

def detect_all_faces(image):
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        return [(x, y, x + w, y + h) for x, y, w, h in faces]
    except:
        return []

def apply_face_mask(source, generated):
    x1, y1, x2, y2 = detect_face(source)
    mask = Image.new("L", source.size, 0)
    draw = ImageDraw.Draw(mask)
    r = min(x2 - x1, y2 - y1) // 2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(20))
    blended = Image.composite(source, generated, mask)
    var = np.array(source.convert('L')).var()
    alpha = max(0.7, min(0.95, 0.8 - var / 10000))
    return Image.blend(generated, blended, alpha)

def enhance_with_codeformer(face_img, fidelity=0.5):
    face_np = np.array(face_img)
    t_orig = torch.from_numpy(face_np).permute(2, 0, 1).float() / 255.0
    t_orig = t_orig.unsqueeze(0).to(DEVICE)
    sm_np = cv2.GaussianBlur(face_np, (5, 5), sigmaX=1.0)
    sm = torch.from_numpy(sm_np).permute(2, 0, 1).float() / 255.0
    sm = sm.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        t = sm + (t_orig - sm) * 1.5
        r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        r *= 1 + 0.05 * fidelity
        g *= 1 + 0.03 * fidelity
        b *= 0.98 - 0.02 * fidelity
        t = torch.cat([r, g, b], dim=1).clamp(0, 1)
    out = (t * 255.0).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(out)

def esrgan_upscale(img, scale=4):
    up = img.resize((img.width * scale, img.height * scale), resample=Image.LANCZOS)
    up = ImageEnhance.Sharpness(up).enhance(1.3)
    up = up.filter(ImageFilter.SMOOTH_MORE)
    return up

def professional_upscale(img, scale=4, face_enhance=True):
    up = esrgan_upscale(img, scale)
    if not face_enhance:
        return up
    try:
        for box in detect_all_faces(up):
            x1, y1, x2, y2 = box
            pad = int(min(x2 - x1, y2 - y1) * 0.1)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(up.width, x2 + pad), min(up.height, y2 + pad)
            face = up.crop((x1, y1, x2, y2))
            face = enhance_with_codeformer(face, fidelity=FACE_ENHANCE_STRENGTH)
            mask = Image.new('L', face.size, 255).filter(ImageFilter.GaussianBlur(20))
            up.paste(face, (x1, y1), mask)
    except Exception as e:
        print(f"⚠️ Face enhance failed: {e}")
    return up

def advanced_upscale(img, scale=4):
    up = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    up = ImageEnhance.Sharpness(up).enhance(1.2)
    up = up.filter(ImageFilter.SMOOTH_MORE)
    return ImageEnhance.Contrast(up).enhance(1.05)

def genai_enhance_image_api_method(image_path):
    # Prevent concurrent requests to avoid memory conflicts
    if not _processing_lock.acquire(blocking=False):
        raise RuntimeError("Another enhancement is already in progress. Please wait and try again.")
    
    try:
        # Debug: Log model versions and device
        import diffusers, transformers
        print(f"[GENAI DEBUG] torch: {torch.__version__}, diffusers: {diffusers.__version__}, transformers: {transformers.__version__}, device: {DEVICE}")
        print(f"[GENAI DEBUG] Using SD model: {pipe.__class__.__name__}")
        print(f"[GENAI DEBUG] Using CLIP model: {clip_model.__class__.__name__}")
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        print(f"[GENAI DEBUG] Input image size: {img.size}, mode: {img.mode}")
        # Save input image for comparison
        img.save("debug_genai_input.jpg")
        
        # Prepare CLIP embeddings
        ti = clip_proc(text=[PROMPT], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_emb = clip_model.get_text_features(**ti)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        # ── MAIN PIPELINE: CLIP + SD Grid Search + Upscale (EXACT COPY FROM STANDALONE) ──
        print("🚀 Starting grid search...")
        best_score, best_params = -1.0, None

        for strength, gs in itertools.product(STRENGTHS, GUIDANCE_SCALES):
            print(f" • strength={strength:.2f}, guidance_scale={gs:.1f}")
            init = img.resize(LOWRES)
            out = pipe(
                prompt=PROMPT,
                negative_prompt=NEG_PROMPT,
                image=init,
                strength=strength,
                guidance_scale=gs,
                num_inference_steps=40,
                generator=torch.Generator(DEVICE).manual_seed(42)
            ).images[0].resize(LOWRES)

            out = apply_face_mask(init, out)
            ci = clip_proc(images=out, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                ie = clip_model.get_image_features(**ci)
                ie = ie / ie.norm(dim=-1, keepdim=True)
            img_score = float(cosine_similarity(text_emb.cpu().numpy(), ie.cpu().numpy())[0,0])
            sharp = np.array(out).std()
            fft = np.fft.fftshift(np.fft.fft2(np.array(out).mean(axis=2)))
            hf = (20 * np.log(np.abs(fft))[10:-10,10:-10]).mean()
            total = 0.6 * img_score + 0.2 * (sharp / 30) + 0.2 * (hf / 30)
            print(f"   → score={total:.3f}")
            if total > best_score:
                best_score, best_params = total, (strength, gs)
                print("   💾 new best")
            clear_memory()

        print(f"✅ Best params: {best_params}")
        print("🎨 Generating high-res SD output...")

        s, g = best_params
        hr = img.resize(HIGHRES)
        final = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=hr,
            strength=s,
            guidance_scale=g,
            num_inference_steps=248,
            generator=torch.Generator(DEVICE).manual_seed(42)
        ).images[0]
        final = apply_face_mask(hr, final)
        print(f"→ Base output generated")

        print("⚡ Applying upscaling...")
        try:
            final_hr = professional_upscale(final, scale=4, face_enhance=True)
        except Exception:
            print("⚠️ professional_upscale failed; using advanced_upscale")
            final_hr = advanced_upscale(final, scale=4)
        print(f"✨ Final high-res upscaling complete")

        # Clear memory before IP-Adapter stage
        clear_memory()

        # ── IP-ADAPTER STAGE (REUSE EXISTING PIPELINE) ────────────────────────────────
        print("🔗 Using existing IP-Adapter pipeline...")
        
        print("📐 Preparing IP input image...")
        init_ip = final_hr.resize((512, 512))

        print("🚀 Running IP-Adapter img2img...")
        result = ip_pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=init_ip,
            ip_adapter_image=init_ip,
            strength=s,
            guidance_scale=g,
            num_inference_steps=148,
        ).images[0]
        print(f"✅ IP-Adapter stage complete")

        # ── FINAL IP-ADAPTER REFINEMENT (EXACT COPY FROM STANDALONE) ─────────────────────
        print("🔁 Running final IP-Adapter img2img refinement...")

        FINAL_INIT_RES = (512, 512)
        FINAL_STRENGTH = 0.6
        FINAL_GUIDANCE_SCALE = 8.0
        FINAL_STEPS = 148
        FINAL_NUM_ITERS = 1

        img_final = result.resize(FINAL_INIT_RES, Image.LANCZOS)
        for i in range(1, FINAL_NUM_ITERS + 1):
            print(f"▶ Iteration {i}/{FINAL_NUM_ITERS} ...", end=" ")
            with torch.autocast(DEVICE):
                out = ip_pipe(
                    prompt=PROMPT,
                    negative_prompt=NEG_PROMPT,
                    image=img_final,
                    ip_adapter_image=img_final,
                    strength=FINAL_STRENGTH,
                    guidance_scale=FINAL_GUIDANCE_SCALE,
                    num_inference_steps=FINAL_STEPS,
                    generator=torch.Generator(DEVICE).manual_seed(42 + i)
                ).images[0]
            img_final = out
            print("done")
        
        # Generate the "grid search result" at high-res for display purposes
        # (since standalone doesn't return grid search result, we'll create one at high-res)
        print("🎨 Generating high-res grid search representation...")
        grid_result = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=hr,
            strength=s,
            guidance_scale=g,
            num_inference_steps=40,  # Same steps as grid search
            generator=torch.Generator(DEVICE).manual_seed(42)
        ).images[0]
        grid_result = apply_face_mask(hr, grid_result)

        # Final memory cleanup
        clear_memory()

        # Save all outputs for debug
        grid_result.save("debug_genai_best_grid.jpg")
        final.save("debug_genai_highres.jpg")
        final_hr.save("debug_genai_upscaled.jpg")
        img_final.save("debug_genai_ipadapter.jpg")
        
        # Return all 4 images (same as what the standalone script generates)
        return [grid_result, final, final_hr, img_final]
        
    finally:
        # Always release the lock
        _processing_lock.release() 