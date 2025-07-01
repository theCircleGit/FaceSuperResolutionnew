#!/usr/bin/env python3
"""
Super Resolution Web Application Startup Script
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'flask_cors', 'pillow', 'numpy', 'cv2', 
        'tensorflow', 'torch', 'torchvision', 'skimage', 
        'scipy', 'dlib', 'basicsr'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'skimage':
                import skimage
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if required model files exist"""
    model_files = [
        'weights/FaceESRGAN/90000_G.pth',
        'weights/CodeFormer/codeformer.pth',
        'weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat'
    ]
    
    missing_files = []
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Warning: Missing model files: {', '.join(missing_files)}")
        print("The application may not work properly without these files.")
        print("Please download the required model weights.")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'weights/FaceESRGAN', 'weights/CodeFormer', 'weights/dlib']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main startup function"""
    print("🚀 Starting Super Resolution Web Application...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\n📁 Creating directories...")
    create_directories()
    
    print("\n🤖 Checking model files...")
    check_model_files()
    
    print("\n" + "=" * 50)
    print("🎉 All checks completed! Starting the application...")
    print("🌐 The application will be available at: http://localhost:5000")
    print("📱 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 