# Gen AI Enhancement System - Updated 13/7/25

This system provides advanced AI-powered image enhancement using Stable Diffusion with IP-Adapter for face preservation and high-resolution upscaling.

## 🚀 Features

- **Stable Diffusion + IP-Adapter**: Advanced image-to-image generation with face preservation
- **Robust Face Detection**: Multiple fallback methods for reliable face detection
- **Deblurring**: Automatic blur removal using Unsharp Masking
- **Skin Smoothing**: Intelligent skin texture enhancement
- **High-Resolution Upscaling**: 4x upscaling with face-specific enhancement
- **Quality Assessment**: Comprehensive scoring system for result comparison
- **Final Refinement**: Additional refinement pass for very blurry/pixelated images

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- 4GB+ VRAM for GPU processing

### Dependencies
All dependencies are automatically installed via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages:
- `torch>=2.0.1`
- `diffusers>=0.21.0`
- `transformers>=4.30.0`
- `accelerate>=0.20.0`
- `opencv-python>=4.8.1`
- `scikit-image>=0.21.0`
- `Pillow>=10.0.1`

## 🛠️ Setup

### 1. Install Dependencies
```bash
cd Super-Resolution-main
pip install -r requirements.txt
```

### 2. Download Face Detection Models
```bash
python download_face_models.py
```

This will download:
- `deploy.prototxt.txt` - Face detection model configuration
- `res10_300x300_ssd_iter_140000.caffemodel` - Face detection model weights

### 3. Verify IP-Adapter Model
Ensure the IP-Adapter model is in the correct location:
```
models/ip-adapter-plus-face_sd15.bin
```

## 🎯 Usage

### Main Enhancement Pipeline

The main enhancement is handled by `genai_enhance.py`:

```python
from genai_enhance import genai_enhance_image_api_method

# Enhance an image
result = genai_enhance_image_api_method("input_image.jpg")

# Access results
enhanced_images = result['images']  # List of 4 enhanced versions
recommended_idx = result['recommended_idx']  # Best version index
```

### Output Files

The system generates multiple output files:

1. **`output_1.png`** - Base Stable Diffusion output (deblurred)
2. **`output_2.png`** - High-resolution upscaled version
3. **`output_3.png`** - IP-Adapter enhanced version
4. **`output_4.png`** - Final refined version (best quality)

### Final Refinement (Optional)

For very blurry or highly pixelated images, run the final refinement:

```bash
python final_refinement.py
```

This takes `output_3.png` and creates an additional refined `output_4.png`.

## 🔧 Configuration

### Key Parameters (in `genai_enhance.py`)

```python
# Generation parameters
STRENGTHS = [0.25, 0.35, 0.45]        # Denoising strength range
GUIDANCE_SCALES = [5.0, 7.0, 9.0]     # Guidance scale range
LOWRES = (320, 320)                   # Low-res processing size
HIGHRES = (768, 768)                  # High-res processing size
FACE_ENHANCE_STRENGTH = 0.7           # Face enhancement intensity

# Prompts
PROMPT = "ultra-high-definition portrait, smooth skin texture..."
NEG_PROMPT = "pixelated, jagged edges, grainy, noisy..."
```

### Final Refinement Parameters (in `final_refinement.py`)

```python
STRENGTH = 0.6           # Denoising strength
GUIDANCE_SCALE = 8.0     # Guidance scale
STEPS = 148              # Inference steps
NUM_ITERS = 1            # Number of refinement iterations
```

## 📊 Quality Assessment

The system provides comprehensive quality scoring:

- **Similarity Scores**: How well the enhanced image preserves the original
- **Quality Scores**: Visual quality metrics (sharpness, clarity)
- **Combined Scores**: Weighted combination of similarity and quality
- **Recommended Index**: Automatically selects the best version

## 🔍 Debug Outputs

The system saves debug images for analysis:

- `debug_genai_input.jpg` - Original input image
- `debug_genai_best_grid.jpg` - Grid search result
- `debug_genai_highres.jpg` - High-res SD output
- `debug_genai_upscaled.jpg` - Upscaled version
- `debug_genai_ipadapter.jpg` - IP-Adapter final result

## ⚡ Performance Tips

### GPU Optimization
- Use CUDA for significantly faster processing
- Ensure sufficient VRAM (4GB+ recommended)
- Close other GPU applications during processing

### Memory Management
- The system automatically clears GPU memory between stages
- For large images, consider reducing `HIGHRES` resolution
- Use `clear_memory()` function if needed

### Processing Time
- **Grid Search**: ~2-3 minutes (9 parameter combinations)
- **High-Res Generation**: ~1-2 minutes
- **Upscaling**: ~30 seconds
- **IP-Adapter**: ~2-3 minutes
- **Total**: ~6-10 minutes per image

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'flask'"**
   ```bash
   pip install Flask==2.3.3
   ```

2. **Face detection fails**
   - Run `python download_face_models.py`
   - System will fall back to Haar cascade detection

3. **Out of memory errors**
   - Reduce `HIGHRES` resolution
   - Close other applications
   - Use CPU processing if GPU memory is insufficient

4. **IP-Adapter not found**
   - Ensure `models/ip-adapter-plus-face_sd15.bin` exists
   - Download from Hugging Face if missing

### Error Recovery

The system includes robust error handling:
- Face detection fallbacks
- Memory cleanup between stages
- Graceful degradation for missing models

## 📈 Results

The system typically produces:
- **4x resolution increase** (e.g., 320x320 → 1280x1280)
- **Improved facial details** with preserved identity
- **Enhanced skin texture** and complexion
- **Reduced noise and artifacts**
- **Professional studio quality** appearance

## 🔄 Integration

### Flask Web Application
The system integrates with the existing Flask web app in `demo.py`:

```python
# In demo.py, the genai_enhance_image_api_method is called
# to process uploaded images through the web interface
```

### API Usage
```python
# Direct API call
result = genai_enhance_image_api_method("path/to/image.jpg")
best_image = result['images'][result['recommended_idx']]
best_image.save("best_result.png")
```

## 📝 Changelog

### Version 13/7/25
- ✅ Added deblurring functionality
- ✅ Improved face detection with DNN fallback
- ✅ Enhanced skin smoothing algorithms
- ✅ Added final refinement pipeline
- ✅ Updated prompts for better results
- ✅ Improved memory management
- ✅ Added comprehensive quality assessment

## 🤝 Contributing

To contribute to the gen AI enhancement system:

1. Test with various image types and qualities
2. Experiment with different prompts and parameters
3. Report issues with specific error messages
4. Suggest improvements to the enhancement pipeline

## 📄 License

This enhancement system is part of the Face Super Resolution project. 