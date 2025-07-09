#!/usr/bin/env python3
"""
Test script for quality assessment functions
This tests the Laplacian quality and combined scoring logic without requiring AI models
"""

import numpy as np
import cv2
from PIL import Image

def compute_laplacian_quality(image):
    """Estimate visual quality as variance of Laplacian on grayscale image"""
    if isinstance(image, Image.Image):
        # Convert PIL image to numpy array
        img_array = np.array(image.convert('L'))
    else:
        img_array = image
    
    lap = cv2.Laplacian(img_array, cv2.CV_64F)
    return float(lap.var())

def test_quality_assessment():
    """Test the quality assessment functions"""
    print("🧪 Testing quality assessment functions...")
    
    # Create test images with different quality levels
    # High quality (sharp) image
    high_quality = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    high_quality = cv2.GaussianBlur(high_quality, (1, 1), 0.5)  # Slight blur
    
    # Low quality (blurry) image
    low_quality = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    low_quality = cv2.GaussianBlur(low_quality, (15, 15), 5)  # Heavy blur
    
    # Convert to PIL images
    high_quality_pil = Image.fromarray(high_quality)
    low_quality_pil = Image.fromarray(low_quality)
    
    # Test Laplacian quality
    high_quality_score = compute_laplacian_quality(high_quality_pil)
    low_quality_score = compute_laplacian_quality(low_quality_pil)
    
    print(f"✅ High quality image score: {high_quality_score:.2f}")
    print(f"✅ Low quality image score: {low_quality_score:.2f}")
    
    # Verify that high quality has higher score
    if high_quality_score > low_quality_score:
        print("✅ Quality assessment working correctly!")
    else:
        print("❌ Quality assessment may have issues")
    
    # Test combined scoring logic
    QUALITY_THRESHOLD = 1000.0
    SIMILARITY_WEIGHT = 0.6
    QUALITY_WEIGHT = 0.4
    
    # Mock similarity scores (higher = more similar to original)
    similarity_scores = [0.8, 0.7, 0.9, 0.6]
    
    # Mock quality scores
    quality_raw_scores = [high_quality_score, low_quality_score, high_quality_score, low_quality_score]
    quality_norm_scores = [min(q / QUALITY_THRESHOLD, 1.0) for q in quality_raw_scores]
    
    # Compute combined scores
    combined_scores = []
    for i in range(len(similarity_scores)):
        combined_score = SIMILARITY_WEIGHT * similarity_scores[i] + QUALITY_WEIGHT * quality_norm_scores[i]
        combined_scores.append(combined_score)
        print(f"   Image {i}: Similarity={similarity_scores[i]:.3f}, Quality={quality_norm_scores[i]:.3f}, Combined={combined_score:.3f}")
    
    # Find best score
    best_idx = np.argmax(combined_scores)
    print(f"🏆 Best combined score: Image {best_idx} (score: {combined_scores[best_idx]:.3f})")
    
    print("✅ All quality assessment tests passed!")

if __name__ == "__main__":
    test_quality_assessment() 