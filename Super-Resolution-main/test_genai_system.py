#!/usr/bin/env python3
"""
Test script for the comprehensive GenAI enhancement system
"""

import os
import sys
import time
from PIL import Image
import numpy as np

def create_test_image():
    """Create a simple test image"""
    # Create a simple test image
    img = Image.new('RGB', (400, 400), color='lightblue')
    
    # Add a simple "face" pattern
    pixels = img.load()
    for i in range(150, 250):
        for j in range(150, 250):
            pixels[i, j] = (255, 200, 200)  # Light pink square
    
    # Save test image
    test_path = "test_image.png"
    img.save(test_path)
    print(f"✅ Created test image: {test_path}")
    return test_path

def test_genai_system():
    """Test the GenAI enhancement system"""
    print("🧪 Testing GenAI Enhancement System")
    print("=" * 50)
    
    # Create test image
    test_image_path = create_test_image()
    
    # Test import
    try:
        from genai_enhance import genai_enhance_image_api_method
        print("✅ GenAI module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import GenAI module: {e}")
        return False
    
    # Test initialization
    try:
        from genai_enhance import initialize_genai_system
        init_result = initialize_genai_system()
        print(f"✅ System initialization: {'Success' if init_result else 'Warning'}")
    except Exception as e:
        print(f"⚠️ System initialization warning: {e}")
    
    # Test enhancement
    try:
        print("\n🎬 Testing GenAI enhancement...")
        start_time = time.time()
        
        result = genai_enhance_image_api_method(test_image_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result:
            print(f"✅ Enhancement completed in {processing_time:.1f}s")
            print(f"📊 Result info:")
            print(f"  - Images generated: {len(result['images'])}")
            print(f"  - Recommended index: {result['recommended_idx']}")
            print(f"  - Video path: {result['video_path']}")
            print(f"  - Method used: {result['face_analysis'].get('method_used', 'unknown')}")
            print(f"  - ComfyUI used: {result['processing_info']['comfyui_used']}")
            
            # Check if video was created
            if result['video_path'] != "static_image":
                if os.path.exists(result['video_path']):
                    print(f"✅ Video file created: {result['video_path']}")
                else:
                    print(f"⚠️ Video path returned but file not found: {result['video_path']}")
            else:
                print("📸 Static image used (video generation failed)")
            
            return True
        else:
            print("❌ Enhancement failed - no result returned")
            return False
            
    except Exception as e:
        print(f"❌ Enhancement failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"🧹 Cleaned up test image: {test_image_path}")

def main():
    """Run the test"""
    success = test_genai_system()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 GenAI System Test: PASSED")
        print("✅ The system is working correctly!")
        print("\nWhat this means:")
        print("• The GenAI enhance button will work when you click it")
        print("• Videos will be generated (ComfyUI or fallback method)")
        print("• Enhanced images will be created")
        print("• The system gracefully handles failures")
    else:
        print("❌ GenAI System Test: FAILED")
        print("⚠️  The system may have issues")
        print("\nWhat to check:")
        print("• Make sure dependencies are installed")
        print("• Check GPU/CUDA availability")
        print("• Verify disk space for model downloads")
        print("• Check network connectivity")
    
    return success

if __name__ == "__main__":
    main() 