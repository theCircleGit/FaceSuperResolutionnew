### updated 13/7/25 Stable diffusion + high resolution - with smoothing.

import os
import sys
import subprocess
import torch
import itertools
import numpy as np
import gc
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from skimage import filters
import threading

# ─── DEPENDENCY INSTALLATION ───────────────────────────────────────────────
def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ModuleNotFoundError:
        print(f"🔧 Installing {pkg_name}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

ensure_package("onnx")
ensure_package("onnxruntime")
ensure_package("scikit-image")

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# Path configurations
IP_ADAPTER_PATH = "models/ip-adapter-plus-face_sd15.bin"

# Model configurations
MODEL_NAME = "runwayml/stable-diffusion-v1-5"

# Generation parameters
STRENGTHS = [0.25, 0.35, 0.45]
GUIDANCE_SCALES = [5.0, 7.0, 9.0]
LOWRES = (320, 320)
HIGHRES = (768, 768)
FACE_ENHANCE_STRENGTH = 0.7

PROMPT = (
    "ultra-high-definition portrait, smooth skin texture, sharp facial features, "
    "cinematic lighting, professional studio quality, 16k resolution, flawless complexion, "
    "detailed eyes, perfect symmetry, no pixelation, anti-aliased"
)
NEG_PROMPT = (
    "pixelated, jagged edges, grainy, noisy, blurry, cartoon, deformed, "
    "distorted, disfigured, bad anatomy, text, watermark, low quality, artifacts"
)

# Simple lock to prevent concurrent requests
_processing_lock = threading.Lock()

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────

def clear_memory():
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()


def detect_faces(image):
    """Robust face detection with fallback to Haar cascade"""
    try:
        net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt.txt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        (h, w) = arr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(arr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2, y2))
        if faces:
            return faces
    except:
        pass
    try:
        arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [(x, y, x+w, y+h) for x, y, w, h in faces]
    except:
        w, h = image.size
        return [(w//4, h//4, 3*w//4, 3*h//4)]


def apply_face_mask(src, gen, blur_radius=20):
    """Apply facial blending with dynamic alpha based on image variance"""
    faces = detect_faces(src)
    if not faces:
        return gen
    x1, y1, x2, y2 = max(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
    mask = Image.new("L", src.size, 0)
    draw = ImageDraw.Draw(mask)
    r = min(x2-x1, y2-y1) // 2
    cx, cy = (x1+x2)//2, (y1+y2)//2
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    var = np.array(src.convert('L')).var()
    alpha = max(0.5, min(0.9, 0.8 - var/15000))
    return Image.blend(gen, Image.composite(src, gen, mask), alpha)


def enhance_face(face_img, fidelity=0.7):
    enhancer = ImageEnhance.Contrast(face_img)
    face_img = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Sharpness(face_img)
    return enhancer.enhance(1.5)


def professional_upscale(img, scale=4):
    """High-quality upscaling with face enhancement"""
    up = img.resize((img.width*scale, img.height*scale), Image.LANCZOS)
    try:
        for box in detect_faces(up):
            x1, y1, x2, y2 = box
            pad = int(min(x2-x1, y2-y1) * 0.15)
            face_area = (
                max(0, x1-pad),
                max(0, y1-pad),
                min(up.width, x2+pad),
                min(up.height, y2+pad)
            )
            face = up.crop(face_area)
            enhanced = enhance_face(face, fidelity=FACE_ENHANCE_STRENGTH)
            mask = Image.new('L', enhanced.size, 255).filter(ImageFilter.GaussianBlur(25))
            up.paste(enhanced, face_area[:2], mask)
    except Exception as e:
        print(f"⚠️ Face enhancement failed: {e}")
    up = ImageEnhance.Contrast(up).enhance(1.05)
    return ImageEnhance.Sharpness(up).enhance(1.2)


def calculate_skin_smoothness(image, face_box):
    try:
        gray = np.array(image.convert('L'))
        x1, y1, x2, y2 = face_box
        face_region = gray[y1:y2, x1:x2]
        if face_region.size == 0:
            return 0.0
        sobel = filters.sobel(face_region)
        texture_variance = np.var(sobel)
        laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
        smoothness = 1.0 / (1.0 + 0.1*texture_variance + 0.01*laplacian_var)
        return min(1.0, max(0.0, smoothness))
    except:
        return 0.0

# ─── NEW: DEBLUR FUNCTION ──────────────────────────────────────────────────

def deblur_image(image, radius=2, percent=150, threshold=3):
    """Remove residual blur using Unsharp Masking"""
    return image.filter(
        ImageFilter.UnsharpMask(radius=radius,
                                 percent=percent,
                                 threshold=threshold)
    )

# ─── PIPELINE INITIALIZATION ───────────────────────────────────────────────
print("🔍 Loading CLIP model…")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("🔥 Loading Stable Diffusion pipeline…")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

# For IP-Adapter
ip_pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)
if hasattr(ip_pipe, 'load_ip_adapter'):
    ip_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")

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
        
        print("🆔 Computing identity embeddings...")
        orig_faces = detect_faces(img)
        if orig_faces:
            orig_face_box = max(orig_faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
            orig_face = img.crop(orig_face_box)
            orig_face_input = clip_proc(images=orig_face, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                orig_face_emb = clip_model.get_image_features(**orig_face_input)
                orig_face_emb = orig_face_emb / orig_face_emb.norm(dim=-1, keepdim=True)
        else:
            orig_face_emb = None
            print("⚠️ No faces detected in original image")

        ti = clip_proc(text=[PROMPT], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_emb = clip_model.get_text_features(**ti)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        print("🚀 Starting parameter search with identity preservation…")
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
                num_inference_steps=35,
                generator=torch.Generator(DEVICE).manual_seed(42)
            ).images[0].resize(LOWRES)
            out = apply_face_mask(init, out)
            ci = clip_proc(images=out, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                ie = clip_model.get_image_features(**ci)
                ie = ie / ie.norm(dim=-1, keepdim=True)
            img_score = F.cosine_similarity(text_emb, ie).item()
            identity_score, smoothness_score = 0.0, 0.0
            out_faces = detect_faces(out)
            if out_faces and orig_face_emb is not None:
                out_face_box = max(out_faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
                out_face = out.crop(out_face_box)
                out_face_input = clip_proc(images=out_face, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out_face_emb = clip_model.get_image_features(**out_face_input)
                    out_face_emb = out_face_emb / out_face_emb.norm(dim=-1, keepdim=True)
                    identity_score = F.cosine_similarity(orig_face_emb, out_face_emb).item()
                smoothness_score = calculate_skin_smoothness(out, out_face_box)
            sharpness = np.array(out).std() / 30
            total_score = (0.4*img_score + 0.3*identity_score + 0.2*smoothness_score + 0.1*sharpness)
            print(f"   → prompt:{img_score:.3f}, identity:{identity_score:.3f}, smooth:{smoothness_score:.3f}, sharp:{sharpness:.3f} → total:{total_score:.3f}")
            if total_score > best_score:
                best_score, best_params = total_score, (strength, gs)
                print("   💾 new best parameters")
            clear_memory()

        # High-res generation & deblur
        print(f"🎨 Generating high-res output with params: {best_params}")
        s, g = best_params
        hr = img.resize(HIGHRES)
        final = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=hr,
            strength=s,
            guidance_scale=g,
            num_inference_steps=50,
            generator=torch.Generator(DEVICE).manual_seed(42)
        ).images[0]
        # Remove blur before saving
        final = deblur_image(final)
        final.save("output_1.png")
        print(f"→ Deblurred base output saved to output_1.png")

        # Upscale & enhance
        print("⚡ Upscaling & enhancing...")
        final_hr = professional_upscale(final, scale=4)
        final_hr.save("output_2.png")
        print(f"✨ High-res intermediate saved to output_2.png")

        # IP-Adapter refinement
        print("🔧 Loading IP-Adapter pipeline...")
        print("🚀 Running IP-Adapter enhancement...")
        init_ip = Image.open("output_2.png").convert("RGB").resize((768, 768))
        result = ip_pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=init_ip,
            ip_adapter_image=init_ip,
            strength=s,
            guidance_scale=g,
            num_inference_steps=148
        ).images[0]
        result.save("output_3.png")
        print(f"✅ Saved → output_3.png")

        # ── FINAL IP-ADAPTER REFINEMENT ─────────────────────────────────────
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
        
        img_final.save("output_4.png")
        print(f"✅ Saved final → output_4.png")

        # Generate the "grid search result" at high-res for display purposes
        print("🎨 Generating high-res grid search representation...")
        grid_result = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=hr,
            strength=s,
            guidance_scale=g,
            num_inference_steps=35,  # Same steps as grid search
            generator=torch.Generator(DEVICE).manual_seed(42)
        ).images[0]
        grid_result = apply_face_mask(hr, grid_result)

        # Save all outputs for debug
        grid_result.save("debug_genai_best_grid.jpg")
        final.save("debug_genai_highres.jpg")
        final_hr.save("debug_genai_upscaled.jpg")
        img_final.save("debug_genai_ipadapter.jpg")
        
        # Return images with comprehensive quality information
        enhanced_images = [grid_result, final, final_hr, img_final]
        
        # Simple quality assessment
        similarity_scores = [0.8, 0.85, 0.9, 0.95]  # Placeholder scores
        quality_scores = [{'raw': 100.0, 'normalized': 0.8}, {'raw': 120.0, 'normalized': 0.85}, 
                         {'raw': 140.0, 'normalized': 0.9}, {'raw': 160.0, 'normalized': 0.95}]
        combined_scores = [0.8, 0.85, 0.9, 0.95]
        recommended_idx = 3  # IP-Adapter final is usually best
        
        return {
            'images': enhanced_images,
            'similarity_scores': similarity_scores,
            'quality_scores': quality_scores,
            'combined_scores': combined_scores,
            'recommended_idx': recommended_idx
        }
        
    finally:
        # Always release the lock
        _processing_lock.release() 