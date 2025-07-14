#!/usr/bin/env python3
"""
Download required face detection models for gen AI enhancement
"""

import os
import urllib.request
import sys

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

def main():
    """Download required face detection models"""
    print("🔧 Setting up face detection models for gen AI enhancement...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Face detection model files
    models = {
        "deploy.prototxt.txt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    success_count = 0
    for filename, url in models.items():
        if not os.path.exists(filename):
            if download_file(url, filename):
                success_count += 1
        else:
            print(f"✅ {filename} already exists")
            success_count += 1
    
    print(f"\n🎉 Setup complete! {success_count}/{len(models)} models ready.")
    
    if success_count < len(models):
        print("⚠️ Some models failed to download. The system will fall back to Haar cascade detection.")
    
    print("\n📝 Note: If face detection models fail to download, the system will automatically")
    print("   fall back to OpenCV's built-in Haar cascade face detection.")

if __name__ == "__main__":
    main() 