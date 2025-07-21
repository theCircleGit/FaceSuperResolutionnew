#!/usr/bin/env python3
"""
Test script for the new GenAI enhancement system
"""

import os
import sys
from PIL import Image
import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_genai_enhancement():
    """Test the new GenAI enhancement system"""
    print("🧪 Testing new GenAI enhancement system...")
    
    try:
        # Import the new GenAI module
        from genai_enhance import genai_enhance_image_api_method
        
        # Check if we have any test images
        test_images = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            test_images.extend([f for f in os.listdir('.') if f.lower().endswith(f'.{ext}')])
        
        if not test_images:
            print("❌ No test images found. Please add a test image to the directory.")
            return False
        
        # Use the first available image
        test_image = test_images[0]
        print(f"📸 Testing with image: {test_image}")
        
        # Test the GenAI enhancement
        result = genai_enhance_image_api_method(test_image)
        
        if result is None:
            print("❌ GenAI enhancement returned None")
            return False
        
        # Check the result structure
        required_keys = ['images', 'similarity_scores', 'quality_scores', 'combined_scores', 'recommended_idx']
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            print(f"❌ Missing keys in result: {missing_keys}")
            return False
        
        # Check the results
        images = result['images']
        similarity_scores = result['similarity_scores']
        quality_scores = result['quality_scores']
        recommended_idx = result['recommended_idx']
        
        print(f"✅ Generated {len(images)} enhancement variants")
        print(f"📊 Quality scores: {[f'{score:.3f}' for score in quality_scores]}")
        print(f"🎯 Similarity scores: {[f'{score:.3f}' for score in similarity_scores]}")
        print(f"🏆 Recommended variant: {recommended_idx}")
        
        # Check if face analysis is available
        if 'face_analysis' in result and result['face_analysis']:
            face_info = result['face_analysis']
            print(f"👤 Face analysis: Yaw={face_info['yaw']:.1f}°, Pitch={face_info['pitch']:.1f}°, Front-facing={face_info['is_front_facing']}")
        
        # Save test results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, img in enumerate(images):
            output_path = f"test_genai_result_{i}_{timestamp}.png"
            img.save(output_path)
            print(f"💾 Saved variant {i} to: {output_path}")
        
        print("✅ GenAI enhancement test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️ Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility():
    """Test that the function interface is compatible with Flask app"""
    print("🔗 Testing Flask app compatibility...")
    
    try:
        from genai_enhance import genai_enhance_image_api_method
        
        # Check function signature
        import inspect
        sig = inspect.signature(genai_enhance_image_api_method)
        params = list(sig.parameters.keys())
        
        if len(params) != 1 or params[0] != 'image_path':
            print(f"❌ Function signature mismatch. Expected (image_path), got ({', '.join(params)})")
            return False
        
        print("✅ Function signature is compatible with Flask app")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎭 Testing New GenAI Enhancement System")
    print("=" * 50)
    
    # Test compatibility first
    if not test_compatibility():
        print("❌ Compatibility test failed - Flask app may not work correctly")
        return
    
    # Test the actual functionality
    if not test_genai_enhancement():
        print("❌ GenAI enhancement test failed")
        return
    
    print("\n🎉 All tests passed! The new GenAI system is ready to use.")
    print("\nYou can now:")
    print("1. Keep using the regular 'Enhance Image' button as before")
    print("2. Use the new 'Enhance with GENAI' button for advanced AI enhancement")
    print("3. The Flask app should work without any changes")

if __name__ == "__main__":
    main() 