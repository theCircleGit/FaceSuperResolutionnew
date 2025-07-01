// Super Resolution App JavaScript

class SuperResolutionApp {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.enhanceCard = document.getElementById('enhanceCard');
        this.enhanceBtn = document.getElementById('enhanceBtn');
        this.processingCard = document.getElementById('processingCard');
        this.resultsCard = document.getElementById('resultsCard');
        this.originalImage = document.getElementById('originalImage');
        this.enhancedImage = document.getElementById('enhancedImage');
        this.errorAlert = document.getElementById('errorAlert');
        this.errorMessage = document.getElementById('errorMessage');
        
        this.currentFile = null;
        this.originalImageData = null;
        this.enhancedImageData = null;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File input change
        this.imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                this.handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        // Click to upload
        document.getElementById('chooseFileBtn').addEventListener('click', () => {
            this.imageInput.value = '';
            this.imageInput.click();
        });
    }

    handleFileUpload(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid image file (JPG, PNG, GIF, BMP)');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB');
            return;
        }

        this.currentFile = file;
        this.displayOriginalImage(file);
        this.showEnhanceCard();
        this.hideResults();
        this.hideError();
    }

    displayOriginalImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.originalImageData = e.target.result;
            this.originalImage.src = this.originalImageData;
        };
        reader.readAsDataURL(file);
    }

    showEnhanceCard() {
        this.enhanceCard.classList.remove('d-none');
        this.enhanceCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    hideResults() {
        this.resultsCard.classList.add('d-none');
    }

    async enhanceImage() {
        if (!this.currentFile) {
            this.showError('Please upload an image first');
            return;
        }

        try {
            this.showProcessing();
            this.hideError();

            const formData = new FormData();
            formData.append('image', this.currentFile);

            const response = await fetch('/api/enhance', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.displayResult(data.enhanced_images);
                // Reset file input so user can upload a new image after enhancing
                this.imageInput.value = '';
                this.currentFile = null;
            } else {
                this.showError(data.error || 'An error occurred while processing the image');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.hideProcessing();
        }
    }

    displayResult(enhancedImages) {
        // Use the balanced enhancement (index 3, which corresponds to fidelity 0.5)
        const enhancedImageData = enhancedImages[3];
        
        if (enhancedImageData) {
            this.enhancedImageData = enhancedImageData;
            this.enhancedImage.src = enhancedImageData;
            
            this.resultsCard.classList.remove('d-none');
            this.enhanceCard.classList.add('d-none');
            this.scrollToResults();
        } else {
            this.showError('Failed to generate enhanced image');
        }
    }

    downloadEnhanced() {
        if (this.enhancedImageData) {
            const link = document.createElement('a');
            link.href = this.enhancedImageData;
            link.download = 'enhanced_image.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    openImageModal(imageData, label) {
        const modal = new bootstrap.Modal(document.getElementById('imageModal'));
        const modalImage = document.getElementById('modalImage');
        const downloadBtn = document.getElementById('downloadBtn');
        
        modalImage.src = imageData;
        downloadBtn.href = imageData;
        downloadBtn.download = `enhanced_${label.replace(/[^a-zA-Z0-9]/g, '_')}.png`;
        
        modal.show();
    }

    showProcessing() {
        this.processingCard.classList.remove('d-none');
        this.resultsCard.classList.add('d-none');
        this.hideError();
    }

    hideProcessing() {
        this.processingCard.classList.add('d-none');
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorAlert.classList.remove('d-none');
        this.hideProcessing();
    }

    hideError() {
        this.errorAlert.classList.add('d-none');
    }

    scrollToResults() {
        setTimeout(() => {
            this.resultsCard.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 100);
    }

    generateReport() {
        if (!this.originalImageData || !this.enhancedImageData) {
            this.showError('No image data available for report generation');
            return;
        }

        try {
            // Show loading state
            const reportBtn = document.querySelector('button[onclick="app.generateReport()"]');
            const originalText = reportBtn.innerHTML;
            reportBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
            reportBtn.disabled = true;

            // Prepare data for report
            const reportData = {
                original_image: this.originalImageData,
                enhanced_image: this.enhancedImageData,
                filename: this.currentFile ? this.currentFile.name : 'enhanced_image'
            };

            // Send request to generate PDF
            fetch('/api/generate-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(reportData)
            })
            .then(response => {
                if (response.ok) {
                    // Download the PDF
                    return response.blob();
                } else {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Failed to generate report');
                    });
                }
            })
            .then(blob => {
                // Create download link for PDF
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `enhancement_report_${reportData.filename}.pdf`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('Error:', error);
                this.showError(error.message || 'Failed to generate report');
            })
            .finally(() => {
                // Restore button state
                reportBtn.innerHTML = originalText;
                reportBtn.disabled = false;
            });
        } catch (error) {
            console.error('Error:', error);
            this.showError('Error generating report');
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SuperResolutionApp();
});

// Add some helpful console messages
console.log('🚀 Super Resolution App loaded successfully!');
console.log('📸 Ready to enhance your images with AI magic!'); 