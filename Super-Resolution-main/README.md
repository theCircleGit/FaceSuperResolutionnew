# AI Super Resolution Web Application

A modern web-based interface for AI-powered image super-resolution using ESRGAN and CodeFormer models. This application can enhance both face images and general images with multiple fidelity levels.

## Features

- 🎨 **Modern Web Interface**: Beautiful, responsive design with drag-and-drop functionality
- 🤖 **Multiple AI Models**: Uses ESRGAN and CodeFormer for different enhancement approaches
- 👤 **Face-Specific Enhancement**: Specialized processing for facial images
- 📊 **Multiple Fidelity Levels**: Choose from different enhancement intensities (0.0 to 1.0)
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- ⚡ **Real-time Processing**: Live feedback during image processing
- 💾 **Easy Download**: Download enhanced images directly from the browser

## Technology Stack

### Backend
- **Flask**: Python web framework
- **TensorFlow**: Deep learning framework for ESRGAN
- **PyTorch**: Deep learning framework for CodeFormer
- **OpenCV**: Image processing
- **dlib**: Face detection and landmark extraction

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript (ES6+)**: Interactive functionality
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icons

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Git

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd Super-Resolution
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model Weights

You'll need to download the pre-trained model weights. Create the following directory structure:

```
weights/
├── FaceESRGAN/
│   └── 90000_G.pth
├── CodeFormer/
│   └── codeformer.pth
└── dlib/
    └── shape_predictor_68_face_landmarks-fbdc2cb8.dat
```

**Note**: You'll need to obtain these model files from their respective sources or repositories.

### Step 5: Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Web Interface

1. **Upload Image**: Drag and drop an image or click to browse
2. **Processing**: Wait for the AI models to process your image
3. **View Results**: See multiple enhanced versions with different fidelity levels
4. **Download**: Click on any result to download the enhanced image

### API Endpoints

#### POST `/api/enhance`
Enhance an uploaded image.

**Request**: Multipart form data with image file
**Response**: JSON with base64-encoded enhanced images

```json
{
  "success": true,
  "enhanced_images": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    ...
  ],
  "message": "Image enhanced successfully!"
}
```

#### GET `/api/health`
Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "message": "Super Resolution API is running"
}
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)

**Maximum file size**: 16MB

## Enhancement Models

### CodeFormer
- **Fidelity 0.0**: Maximum enhancement, may alter facial features
- **Fidelity 0.2**: High enhancement with some feature preservation
- **Fidelity 0.5**: Balanced enhancement and preservation
- **Fidelity 0.7**: Conservative enhancement with good preservation
- **Fidelity 1.0**: Minimal enhancement, maximum feature preservation

### ESRGAN
- Face-specific enhancement using models trained on facial datasets
- General image enhancement for non-face images

## Project Structure

```
Super-Resolution/
├── app.py                 # Flask application
├── super_resol.py         # Core super-resolution logic
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── app.js        # Frontend JavaScript
├── uploads/              # Temporary upload directory
└── weights/              # Model weights directory
    ├── FaceESRGAN/
    ├── CodeFormer/
    └── dlib/
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce image size or use CPU processing
2. **Model Files Missing**: Ensure all required model weights are downloaded
3. **dlib Installation Issues**: On Windows, you may need to install Visual Studio Build Tools

### Performance Tips

- Use GPU acceleration for faster processing
- Process smaller images for quicker results
- Close other GPU-intensive applications

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ESRGAN](https://github.com/xinntao/ESRGAN) - Enhanced Super-Resolution Generative Adversarial Networks
- [CodeFormer](https://github.com/sczhou/CodeFormer) - Robust Face Restoration and Enhancement
- [dlib](http://dlib.net/) - Machine Learning Toolkit
- [Flask](https://flask.palletsprojects.com/) - Web Framework

## Support

For support and questions, please open an issue on the GitHub repository. 