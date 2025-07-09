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

# ── CONFIG ─────────────────────────────────────────────────────────────────
INPUT_IMAGE      = "/home/user/source/repos/FaceSuperResolutionnew/Super-Resolution-main/1.jpg"
OUTPUT_IMAGE     = "output_1.png"
FINAL_HR_IMAGE   = "high_res_intermediate.png"
# IP-Adapter config
IP_INPUT_IMAGE   = FINAL_HR_IMAGE
IP_OUTPUT_IMAGE  = "final_output.png"
IP_ADAPTER_PATH  = "models/ip-adapter-plus-face_sd15.bin"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters for SD grid search
STRENGTHS        = [0.25, 0.35, 0.45]
GUIDANCE_SCALES  = [5.0, 7.0, 9.0]
LOWRES           = (320, 320)
HIGHRES          = (768, 768)

# Prompts
PROMPT = (
    "ultra-high-definition portrait, smooth skin texture, sharp facial features, eyes, nose, "
    "cinematic lighting, professional studio quality, 16k resolution, flawless complexion, "
    "detailed eyes, razor-sharp eyes, perfect symmetry, same age, same gender, no pixelation, anti-aliased, f/1.4 aperture"
)
NEG_PROMPT = (
    "pixelated, jagged edges, grainy, noisy, blurry, cartoon, jpeg artifacts, anime, blocking, "
    "deformed, distorted, disfigured, bad anatomy, text, signature, gender swap, age change, makeup, watermark, low quality, artifacts"
)

# Face enhancement parameters
FACE_ENHANCE_STRENGTH = 0.7  # Balance identity vs. detail

# ── HELPERS ─────────────────────────────────────────────────────────────────
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

# ── MAIN PIPELINE: CLIP + SD Grid Search + Upscale ──────────────────────────
print("🔍 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
ti = clip_proc(text=[PROMPT], return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    text_emb = clip_model.get_text_features(**ti)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

print("🔥 Loading Stable Diffusion Img2Img pipeline...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

print("🚀 Starting grid search...")
best_score, best_params = -1.0, None
img = Image.open(INPUT_IMAGE).convert("RGB")

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
final.save(OUTPUT_IMAGE)
print(f"→ Base output saved to {OUTPUT_IMAGE}")

print("⚡ Applying upscaling...")
try:
    final_hr = professional_upscale(final, scale=4, face_enhance=True)
except Exception:
    print("⚠️ professional_upscale failed; using advanced_upscale")
    final_hr = advanced_upscale(final, scale=4)
final_hr.save(FINAL_HR_IMAGE)
print(f"✨ Final high-res saved to {FINAL_HR_IMAGE}")

# ── IP-ADAPTER STAGE ────────────────────────────────────────────────────────
print("🔧 Loading Stable Diffusion Img2Img pipeline for IP-Adapter...")
ip_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

print("🔗 Attaching IP-Adapter...")
ip_pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name=os.path.basename(IP_ADAPTER_PATH),
)
ip_pipe.set_ip_adapter_scale(1.0)

print("📐 Preparing IP input image...")
init_ip = Image.open(IP_INPUT_IMAGE).convert("RGB").resize((512, 512))

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

result.save(IP_OUTPUT_IMAGE)
print(f"✅ Saved → {IP_OUTPUT_IMAGE}")

# ── FINAL IP-ADAPTER REFINEMENT (output.png) ───────────────────────────────
print("🔁 Running final IP-Adapter img2img refinement for output.png ...")

FINAL_INPUT_IMAGE = IP_OUTPUT_IMAGE
FINAL_OUTPUT_IMAGE = "output.png"
FINAL_INIT_RES = (512, 512)
FINAL_STRENGTH = 0.6
FINAL_GUIDANCE_SCALE = 8.0
FINAL_STEPS = 148
FINAL_NUM_ITERS = 1

img = Image.open(FINAL_INPUT_IMAGE).convert("RGB").resize(FINAL_INIT_RES, Image.LANCZOS)
for i in range(1, FINAL_NUM_ITERS + 1):
    print(f"▶ Iteration {i}/{FINAL_NUM_ITERS} ...", end=" ")
    with torch.autocast(DEVICE):
        out = ip_pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=img,
            ip_adapter_image=img,
            strength=FINAL_STRENGTH,
            guidance_scale=FINAL_GUIDANCE_SCALE,
            num_inference_steps=FINAL_STEPS,
            generator=torch.Generator(DEVICE).manual_seed(42 + i)
        ).images[0]
    img = out
    print("done")
img.save(FINAL_OUTPUT_IMAGE)
print(f"✅ Saved final → {FINAL_OUTPUT_IMAGE}")

# ── QUALITY ASSESSMENT WITH RESNET + LAPLACIAN ─────────────────────────────
print("\n🔍 Running quality assessment...")

# Import additional required modules
from torchvision import models, transforms

# Configuration for quality assessment
QUALITY_THRESHOLD = 1000.0   # Values above this are treated as max quality
SIMILARITY_WEIGHT = 0.6      # Weight for ResNet cosine similarity
QUALITY_WEIGHT = 0.4         # Weight for visual quality

# Define outputs to assess
OUTPUTS_TO_ASSESS = [
    OUTPUT_IMAGE,        # "output_1.png"
    FINAL_HR_IMAGE,      # "high_res_intermediate.png"
    IP_OUTPUT_IMAGE,     # "final_output.png"
    FINAL_OUTPUT_IMAGE   # "output.png"
]

def load_image_for_assessment(image_path, transform, device):
    """Load an image via PIL, apply preprocessing, and return a tensor."""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def get_feature_extractor(device):
    """Return ResNet50 model truncated before the classification head."""
    base = models.resnet50(pretrained=True)
    layers = list(base.children())[:-1]
    model = torch.nn.Sequential(*layers).to(device)
    model.eval()
    return model

def extract_feature(model, image_tensor):
    """Extract and L2-normalize features from an image tensor."""
    with torch.no_grad():
        feat = model(image_tensor)
    feat = feat.view(feat.size(0), -1)
    return F.normalize(feat, p=2, dim=1)

def compute_laplacian_quality(image_path):
    """Estimate visual quality as variance of Laplacian on grayscale image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image for quality assessment: {image_path}")
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())

# Setup for assessment
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load reference image and setup model
img1_t = load_image_for_assessment(INPUT_IMAGE, transform, DEVICE)
model = get_feature_extractor(DEVICE)
f1 = extract_feature(model, img1_t)

# Assess each output
results = []
print("\n=== Quality Assessment Results ===")

for out_path in OUTPUTS_TO_ASSESS:
    try:
        # Load and process output image
        img2_t = load_image_for_assessment(out_path, transform, DEVICE)
        f2 = extract_feature(model, img2_t)

        # Compute ResNet similarity
        similarity = F.cosine_similarity(f1, f2).item()

        # Compute Laplacian quality
        quality_raw = compute_laplacian_quality(out_path)
        quality_norm = min(quality_raw / QUALITY_THRESHOLD, 1.0)

        # Combined score
        combined_score = SIMILARITY_WEIGHT * similarity + QUALITY_WEIGHT * quality_norm

        # Store results
        results.append({
            'path': out_path,
            'similarity': similarity,
            'quality_raw': quality_raw,
            'quality_norm': quality_norm,
            'combined_score': combined_score
        })

        print(f"\n📁 {out_path}")
        print(f"   ResNet similarity: {similarity:.4f}")
        print(f"   Laplacian quality: {quality_raw:.2f}")
        print(f"   Normalized quality: {quality_norm:.4f}")
        print(f"   Combined score: {combined_score:.4f}")

    except Exception as e:
        print(f"\n❌ {out_path}: Error - {e}")

# Find the best output
if results:
    best_result = max(results, key=lambda x: x['combined_score'])
    print(f"\n🏆 RECOMMENDED: {best_result['path']}")
    print(f"   Best combined score: {best_result['combined_score']:.4f}")
    print(f"   (ResNet: {best_result['similarity']:.4f}, Quality: {best_result['quality_norm']:.4f})")

    # Print ranking
    print(f"\n📊 Ranking (highest to lowest score):")
    sorted_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        marker = "🏆" if result == best_result else f"{i}."
        print(f"   {marker} {result['path']}: {result['combined_score']:.4f}")
else:
    print("\n❌ No outputs could be assessed successfully")

print("\n✅ Quality assessment complete!") 