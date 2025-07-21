#!/usr/bin/env python3
"""
Comprehensive AI Video Generation and Face Processing System
Advanced face enhancement, video generation, and face detection capabilities
"""

import os
import sys
import subprocess
import urllib.request
import torch
import numpy as np
from PIL import Image
import gc
import random
import imageio
import glob
import time
import shutil
import cv2
import dlib
import bz2
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    packages = [
        'torch==2.6.0',
        'torchvision==0.21.0',
        'torchsde',
        'av',
        'diffusers',
        'transformers',
        'xformers==0.0.29.post2',
        'accelerate',
        'tqdm',
        'color-matcher',
        'einops',
        'spandrel',
        'opencv-python',
        'dlib',
        'matplotlib',
        'imageio[ffmpeg]'
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package], 
                         check=True, capture_output=True)
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing {package}: {e.stderr.decode().strip() or 'Unknown error'}")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "models/unet",
        "models/vae", 
        "models/clip",
        "models/text_encoders",
        "models/loras",
        "models/diffusion_models",
        "models/clip_vision",
        "input",
        "output"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def download_file(url, filename):
    """Download a file from URL to filename"""
    print(f"📥 Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def model_download(url: str, dest_dir: str, filename: str = None, silent: bool = True) -> str:
    """Download model with aria2c or fallback to urllib"""
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]
            
        full_path = os.path.join(dest_dir, filename)
        
        # Try aria2c first
        try:
            cmd = [
                'aria2c', '--console-log-level=error', '-c', '-x', '16', '-s', '16', '-k', '1M',
                '-d', dest_dir, '-o', filename, url
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            if silent:
                print(f"✓ Downloaded {filename}")
            return filename
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to urllib
            print(f"Aria2c not available, using urllib for {filename}...")
            return download_file(url, full_path) and filename
            
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return None

def download_models():
    """Download all required AI models"""
    print("📥 Downloading AI models...")
    
    models = {
        # Flux models
        "flux1-kontext-dev-Q6_K.gguf": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/flux1-kontext-dev-Q6_K.gguf",
        "ae.sft": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/ae.sft",
        "clip_l.safetensors": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/clip_l.safetensors",
        "t5xxl_fp8_e4m3fn.safetensors": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/t5xxl_fp8_e4m3fn.safetensors",
        "flux_1_turbo_alpha.safetensors": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/flux_1_turbo_alpha.safetensors",
        "Facezoom_Kontext_LoRA.safetensors": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/Facezoom_Kontext_LoRA.safetensors",
        
        # Wan models
        "Wan2.1-VACE-14B-Q8_0.gguf": "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/Wan2.1-VACE-14B-Q8_0.gguf",
        "Wan21_CausVid_14B_T2V_lora_rank32.safetensors": "https://huggingface.co/Isi99999/Wan2.1BasedModels/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "wan_2.1_vae.safetensors": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
    }
    
    # Map models to directories
    model_dirs = {
        "flux1-kontext-dev-Q6_K.gguf": "models/unet",
        "ae.sft": "models/vae",
        "clip_l.safetensors": "models/clip", 
        "t5xxl_fp8_e4m3fn.safetensors": "models/clip",
        "flux_1_turbo_alpha.safetensors": "models/loras",
        "Facezoom_Kontext_LoRA.safetensors": "models/loras",
        "Wan2.1-VACE-14B-Q8_0.gguf": "models/diffusion_models",
        "Wan21_CausVid_14B_T2V_lora_rank32.safetensors": "models/loras",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": "models/text_encoders",
        "wan_2.1_vae.safetensors": "models/vae",
    }
    
    for filename, url in models.items():
        dest_dir = model_dirs.get(filename, "models")
        if not os.path.exists(os.path.join(dest_dir, filename)):
            model_download(url, dest_dir, filename)
        else:
            print(f"✓ {filename} already exists")

def download_dlib_model():
    """Download dlib face landmark model"""
    model_bz2 = "shape_predictor_68_face_landmarks.dat.bz2"
    model_dat = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(model_dat):
        if not os.path.exists(model_bz2):
            print("Downloading dlib shape predictor (~100 MB)...")
            urllib.request.urlretrieve(
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                model_bz2
            )
        print("Decompressing shape predictor...")
        with bz2.BZ2File(model_bz2, "rb") as f_in, open(model_dat, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(model_bz2)
        print("✓ Dlib model ready.")

def clear_memory():
    """Clear GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Clear tensor objects from globals
    for obj in list(globals().values()):
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            del obj
    gc.collect()

def save_as_image(image, filename_prefix, output_dir="output"):
    """Save single frame as PNG image"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.png"
    
    if torch.is_tensor(image):
        frame = (image.cpu().numpy() * 255).astype(np.uint8)
    else:
        frame = (image * 255).astype(np.uint8)
    
    Image.fromarray(frame).save(output_path)
    return output_path

def save_as_mp4(images, filename_prefix, fps, output_dir="output"):
    """Save images as MP4 video"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.mp4"
    
    frames = []
    for img in images:
        if torch.is_tensor(img):
            frame = (img.cpu().numpy() * 255).astype(np.uint8)
        else:
            frame = (img * 255).astype(np.uint8)
        frames.append(frame)
    
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    
    return output_path

def get_euler_angles(rvec, tvec, cam_m, dist):
    """Get Euler angles from rotation vector"""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = np.degrees(np.arctan2(R[2,1], R[2,2]))
    yaw = np.degrees(np.arctan2(-R[2,0], sy))
    roll = np.degrees(np.arctan2(R[1,0], R[0,0]))
    return pitch, yaw, roll

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio for blink detection"""
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    return (A+B)/(2.0*C)

def find_best_front_frame(video_path):
    """Find the best front-facing frame from video"""
    print("Finding best front-facing frame...")
    
    # Ensure dlib model exists
    model_dat = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_dat):
        download_dlib_model()
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_dat)
    
    # 3D model points for solvePnP
    model_pts = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)
    
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_center = np.array([w/2, h/2])
    
    # Camera intrinsics
    focal = w
    cam_m = np.array([[focal, 0, w/2],
                      [0, focal, h/2],
                      [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4,1))
    
    best_frame, best_score = None, -1e9
    
    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        if not faces:
            continue
        
        # Pick the largest face
        face = max(faces, key=lambda r: r.width()*r.height())
        shape = predictor(gray, face)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], float)
        
        # solvePnP for head pose
        img_pts = np.vstack([pts[30], pts[8], pts[36], pts[45], pts[48], pts[54]])
        ok, rvec, tvec = cv2.solvePnP(model_pts, img_pts, cam_m, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        
        pitch, yaw, roll = get_euler_angles(rvec, tvec, cam_m, dist)
        
        # Eye aspect ratio for blink detection
        leftEAR = eye_aspect_ratio(pts[36:42])
        rightEAR = eye_aspect_ratio(pts[42:48])
        ear = (leftEAR + rightEAR) / 2.0
        
        # Image clarity
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Face centering
        face_center = np.array([face.left() + face.width()/2,
                               face.top() + face.height()/2])
        center_offset = np.linalg.norm(face_center - img_center) / (w/2)
        
        # Score frame based on front-facing criteria
        if abs(yaw) <= 15 and abs(pitch) <= 15 and abs(roll) <= 10 and ear >= 0.18:
            score = (clarity 
                    - abs(yaw) * 5
                    - abs(pitch) * 3
                    - abs(roll) * 2
                    - center_offset * 30)
            
            if score > best_score:
                best_score, best_frame = score, frame.copy()
    
    cap.release()
    
    if best_frame is None:
        print("⚠️ No suitable front-facing frame found, using first frame")
        cap = cv2.VideoCapture(video_path)
        ret, best_frame = cap.read()
        cap.release()
    
    return best_frame

def process_image(image_path, output_path=None):
    """Process a single image for face enhancement"""
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/enhanced_{timestamp}.png"
    
    # Basic image enhancement (placeholder - extend with actual AI processing)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply basic enhancement
    enhanced = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
    
    # Save enhanced image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced)
    
    print(f"✅ Enhanced image saved to: {output_path}")
    return output_path

def main():
    """Main function to run the face enhancement system"""
    print("🎭 Starting AI Face Enhancement System")
    print("=" * 50)
    
    # Setup
    setup_directories()
    
    # Check if dependencies need to be installed
    try:
        import torch
        import cv2
        import dlib
        print("✅ Dependencies already installed")
    except ImportError:
        print("📦 Installing dependencies...")
        install_dependencies()
    
    # Download models
    download_models()
    download_dlib_model()
    
    print("\n🎉 Setup complete!")
    print("\nAvailable functions:")
    print("- process_image(image_path): Enhance a single image")
    print("- find_best_front_frame(video_path): Extract best frame from video")
    print("- save_as_mp4(images, filename, fps): Save image sequence as video")
    print("\nExample usage:")
    print("  enhanced_path = process_image('input/photo.jpg')")
    print("  best_frame = find_best_front_frame('input/video.mp4')")
    
    # Interactive mode
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("1. Process image")
        print("2. Extract best frame from video")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                try:
                    result = process_image(image_path)
                    print(f"✅ Image processed successfully: {result}")
                except Exception as e:
                    print(f"❌ Error processing image: {e}")
            else:
                print("❌ Image file not found")
                
        elif choice == '2':
            video_path = input("Enter video path: ").strip()
            if os.path.exists(video_path):
                try:
                    best_frame = find_best_front_frame(video_path)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"output/best_frame_{timestamp}.png"
                    cv2.imwrite(output_path, best_frame)
                    print(f"✅ Best frame saved to: {output_path}")
                except Exception as e:
                    print(f"❌ Error processing video: {e}")
            else:
                print("❌ Video file not found")
                
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 