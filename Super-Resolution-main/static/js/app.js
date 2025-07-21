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
        this.originalImagePath = null;
        this.enhancedImagePaths = null;
        
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

        // Add event listener for GENAI button
        const genaiBtn = document.getElementById('genaiEnhanceBtn');
        if (genaiBtn) {
            genaiBtn.addEventListener('click', () => this.enhanceWithGenai());
        }

        // Add event listener for Case History tab
        const casesTab = document.getElementById('cases-tab');
        if (casesTab) {
            casesTab.addEventListener('click', () => this.loadCaseHistory());
        }
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
        this.showFileSelected(file);
        this.showEnhanceCard();
        this.hideResults();
        this.hideError();
    }

    showFileSelected(file) {
        // Update upload area to show selected file information
        const uploadContent = this.uploadArea.querySelector('.upload-content');
        const fileSize = (file.size / (1024 * 1024)).toFixed(2); // Convert to MB
        const caseIdInput = document.getElementById('caseIdInput');
        const caseId = caseIdInput ? caseIdInput.value.trim() : '';
        
        uploadContent.innerHTML = `
            <div class="text-success mb-3">
                <i class="fas fa-check-circle fa-3x"></i>
            </div>
            <h6 class="text-success mb-2">File Selected Successfully!</h6>
            <div class="file-info bg-light p-3 rounded">
                <div class="row">
                    <div class="col-12">
                        <strong>Filename:</strong> ${file.name}
                    </div>
                    <div class="col-12">
                        <strong>Size:</strong> ${fileSize} MB
                    </div>
                    <div class="col-12">
                        <strong>Type:</strong> ${file.type}
                    </div>
                    ${caseId ? `<div class="col-12">
                        <strong>Case ID:</strong> ${caseId}
                    </div>` : ''}
                </div>
            </div>
            <div class="mt-3">
                <button class="btn btn-outline-secondary btn-sm" id="changeFileBtn" type="button">
                    <i class="fas fa-edit"></i> Change File
                </button>
            </div>
        `;

        // Add event listener for change file button
        const changeFileBtn = document.getElementById('changeFileBtn');
        if (changeFileBtn) {
            changeFileBtn.addEventListener('click', () => {
                this.resetUploadArea();
                this.imageInput.click();
            });
        }
    }

    resetUploadArea() {
        // Reset upload area to original state
        const uploadContent = this.uploadArea.querySelector('.upload-content');
        uploadContent.innerHTML = `
            <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
            <p class="mb-2">Drag and drop your image here or click to browse</p>
            <p class="text-muted small">Supports: JPG, PNG, GIF, BMP (Max: 16MB)</p>
            <input type="file" id="imageInput" accept="image/*" class="d-none">
            <button class="btn btn-primary" id="chooseFileBtn" type="button">
                <i class="fas fa-folder-open"></i> Choose File
            </button>
        `;

        // Re-add event listener for choose file button
        const chooseFileBtn = document.getElementById('chooseFileBtn');
        if (chooseFileBtn) {
            chooseFileBtn.addEventListener('click', () => {
                this.imageInput.value = '';
                this.imageInput.click();
            });
        }
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
            
            // Add case ID if provided
            const caseIdInput = document.getElementById('caseIdInput');
            if (caseIdInput && caseIdInput.value.trim()) {
                formData.append('case_id', caseIdInput.value.trim());
            }

            const response = await fetch('/api/enhance', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.displayResult(data.enhanced_images);
                // Store file paths for report generation
                this.originalImagePath = data.original_image_path;
                this.enhancedImagePaths = data.enhanced_image_paths;
                // Reset file input and upload area so user can upload a new image after enhancing
                this.imageInput.value = '';
                this.currentFile = null;
                this.resetUploadArea();
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

    async enhanceWithGenai() {
        if (!this.currentFile) {
            this.showError('Please upload an image first');
            return;
        }
        const genaiBtn = document.getElementById('genaiEnhanceBtn');
        if (genaiBtn) {
            genaiBtn.disabled = true;
            genaiBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Enhancing...';
        }
        try {
            this.showProcessing();
            this.hideError();
            const formData = new FormData();
            formData.append('image', this.currentFile);
            
            // Add case ID if provided
            const caseIdInput = document.getElementById('caseIdInput');
            if (caseIdInput && caseIdInput.value.trim()) {
                formData.append('case_id', caseIdInput.value.trim());
            }
            
            const response = await fetch('/api/genai-enhance', {
                method: 'POST',
                body: formData
            });
            if (response.status === 429) {
                // Backend is busy, keep spinner/loading and do not show error
                return;
            }
            const data = await response.json();
            if (response.ok && data.success) {
                this.displayGenaiResult(data.enhanced_images, data.similarity_scores, data.recommended_idx);
                this.originalImagePath = data.original_image_path;
                this.enhancedImagePaths = data.enhanced_image_paths;
                
                // Add video display if available
                if (data.video_generated && data.video_url) {
                    this.displayVideoSection(data.video_url);
                }
                
                this.imageInput.value = '';
                this.currentFile = null;
                this.resetUploadArea();
            } else {
                this.showError(data.error || 'An error occurred while processing the image');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.hideProcessing();
            if (genaiBtn) {
                genaiBtn.disabled = false;
                genaiBtn.innerHTML = '<i class="fas fa-robot"></i> Enhance with GENAI';
            }
        }
    }

    displayResult(enhancedImages) {
        // Display all 5 enhanced images with their corresponding fidelity levels
        const imageIds = ['enhancedImage0', 'enhancedImage1', 'enhancedImage2', 'enhancedImage3', 'enhancedImage4'];
        const fidelityLevels = [0.0, 0.7, 0.2, 0.5, 1.0]; // This matches the order returned by the backend
        
        if (enhancedImages && enhancedImages.length >= 5) {
            // Store all enhanced image data
            this.enhancedImageData = enhancedImages;
            
            // Set each enhanced image
            for (let i = 0; i < 5; i++) {
                const imgElement = document.getElementById(imageIds[i]);
                if (imgElement && enhancedImages[i]) {
                    imgElement.src = enhancedImages[i];
                    // Add click event to open in modal
                    imgElement.onclick = () => this.openImageModal(enhancedImages[i], `Fidelity ${fidelityLevels[i]}`);
                }
            }
            
            this.resultsCard.classList.remove('d-none');
            this.enhanceCard.classList.add('d-none');
            this.scrollToResults();
        } else {
            this.showError('Failed to generate enhanced images');
        }
    }

    displayGenaiResult(enhancedImages, similarityScores, recommendedIdx) {
        // For GenAI, we only display one enhanced image (the best frame)
        const imageIds = ['enhancedImage0', 'enhancedImage1', 'enhancedImage2', 'enhancedImage3', 'enhancedImage4'];
        
        if (enhancedImages && enhancedImages.length >= 1) {
            this.enhancedImageData = enhancedImages;
            
            // Show only the first image slot with the enhanced image
            const firstImg = document.getElementById('enhancedImage0');
            const firstLabel = firstImg?.parentElement.querySelector('h6');
            
            if (firstImg && enhancedImages[0]) {
                firstImg.src = enhancedImages[0];
                firstImg.onclick = () => this.openImageModal(enhancedImages[0], "Best Frame (Enhanced)");
                if (firstLabel) firstLabel.textContent = "Best Frame (Enhanced)";
                
                // Show the first image container
                firstImg.parentElement.style.display = 'block';
                
                // Add badge
                let badgeContainer = firstImg?.parentElement.querySelector('.mt-2');
                if (!badgeContainer) {
                    badgeContainer = document.createElement('div');
                    badgeContainer.className = 'mt-2';
                    firstImg.parentElement.appendChild(badgeContainer);
                }
                badgeContainer.innerHTML = '<span class="badge bg-primary">AI Enhanced</span>';
            }
            
            // Hide all other image slots for GenAI results
            for (let i = 1; i < imageIds.length; i++) {
                const imgElement = document.getElementById(imageIds[i]);
                if (imgElement && imgElement.parentElement) {
                    imgElement.parentElement.style.display = 'none';
                }
            }
            
            this.resultsCard.classList.remove('d-none');
            this.enhanceCard.classList.add('d-none');
            this.scrollToResults();
        } else {
            this.showError('Failed to generate GENAI enhanced images');
        }
    }

    displayVideoSection(videoUrl) {
        // Remove existing video section if any
        const existingVideo = document.getElementById('genai-video-section');
        if (existingVideo) {
            existingVideo.remove();
        }

        // Determine file type from URL
        const isGif = videoUrl.toLowerCase().endsWith('.gif');
        const fileType = isGif ? 'GIF Animation' : 'MP4 Video';

        // Create video section
        const videoSection = document.createElement('div');
        videoSection.id = 'genai-video-section';
        videoSection.className = 'mt-4 p-3 border rounded bg-light';

        let mediaHTML;
        if (isGif) {
            // For GIF files, use img tag
            mediaHTML = `
                <img 
                    id="genai-video-player"
                    src="${videoUrl}" 
                    class="w-100 rounded" 
                    style="max-height: 400px; background-color: #000;"
                    onload="console.log('🎬 GIF loaded successfully')"
                    onerror="console.error('🎬 GIF error'); app.handleVideoError(this)">
                <div class="mt-2">
                    <small class="text-muted">GIF URL: <code>${videoUrl}</code></small>
                </div>
            `;
        } else {
            // For MP4 files, use video tag
            mediaHTML = `
                <video 
                    id="genai-video-player"
                    controls 
                    class="w-100 rounded" 
                    style="max-height: 400px; background-color: #000;"
                    preload="metadata"
                    onloadstart="console.log('🎬 Video load started')"
                    onloadeddata="console.log('🎬 Video data loaded')"
                    oncanplay="console.log('🎬 Video can play')"
                    onerror="console.error('🎬 Video error:', this.error); app.handleVideoError(this)">
                    <source src="${videoUrl}" type="video/mp4">
                    <p class="text-muted p-3">Your browser does not support the video tag.</p>
                </video>
                <div class="mt-2">
                    <small class="text-muted">Video URL: <code>${videoUrl}</code></small>
                </div>
            `;
        }

        videoSection.innerHTML = `
            <h5 class="mb-3"><i class="fas fa-video text-primary"></i> Generated AI ${fileType}</h5>
            <div class="row">
                <div class="col-md-8">
                    ${mediaHTML}
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">${fileType} Details</h6>
                            <p class="small text-muted">AI-generated head movement sequence</p>
                            <div class="d-grid gap-2">
                                <a href="${videoUrl}" download="genai_video${isGif ? '.gif' : '.mp4'}" class="btn btn-outline-primary btn-sm">
                                    <i class="fas fa-download"></i> Download ${fileType}
                                </a>
                                <button onclick="app.testVideoUrl('${videoUrl}')" class="btn btn-outline-info btn-sm">
                                    <i class="fas fa-play"></i> Test ${fileType} URL
                                </button>
                                <a href="${videoUrl}" target="_blank" class="btn btn-outline-secondary btn-sm">
                                    <i class="fas fa-external-link-alt"></i> Open in New Tab
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div id="video-error-info" class="alert alert-warning mt-3" style="display: none;">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>${fileType} Playback Issue:</strong> The ${fileType.toLowerCase()} file may not be properly encoded for web browsers.
                You can still <a href="${videoUrl}" download>download the ${fileType.toLowerCase()}</a> to view it locally.
            </div>
        `;

        // Insert after results card
        this.resultsCard.parentNode.insertBefore(videoSection, this.resultsCard.nextSibling);
        
        console.log(`🎬 ${fileType} section added with URL:`, videoUrl);
    }

    handleVideoError(videoElement) {
        console.error('🎬 Video playback error:', videoElement.error);
        const errorInfo = document.getElementById('video-error-info');
        if (errorInfo) {
            errorInfo.style.display = 'block';
        }
        
        // Show specific error message
        if (videoElement.error) {
            console.error('🎬 Error code:', videoElement.error.code);
            console.error('🎬 Error message:', videoElement.error.message);
        }
    }

    testVideoUrl(videoUrl) {
        console.log('🎬 Testing video URL:', videoUrl);
        fetch(videoUrl, { method: 'HEAD' })
            .then(response => {
                console.log('🎬 Video URL response:', response.status, response.statusText);
                console.log('🎬 Content-Type:', response.headers.get('Content-Type'));
                console.log('🎬 Content-Length:', response.headers.get('Content-Length'));
                
                if (response.ok) {
                    this.showNotification('Video URL is accessible!', 'success');
                } else {
                    this.showNotification(`Video URL error: ${response.status}`, 'warning');
                }
            })
            .catch(error => {
                console.error('🎬 Video URL test failed:', error);
                this.showNotification('Video URL test failed', 'danger');
            });
    }

    downloadEnhanced() {
        // Download the balanced enhancement (index 3, fidelity 0.5)
        if (this.enhancedImageData && this.enhancedImageData[3]) {
            const link = document.createElement('a');
            link.href = this.enhancedImageData[3];
            link.download = 'enhanced_image_balanced.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    downloadAllEnhanced() {
        if (this.enhancedImageData && this.enhancedImageData.length >= 5) {
            const fidelityLabels = ['maximum', 'conservative', 'high', 'balanced', 'minimal'];
            const fidelityValues = [0.0, 0.7, 0.2, 0.5, 1.0];
            
            // Download each enhanced image individually
            this.enhancedImageData.forEach((imageData, index) => {
                if (imageData) {
                    setTimeout(() => {
                        const link = document.createElement('a');
                        link.href = imageData;
                        link.download = `enhanced_image_${fidelityLabels[index]}_${fidelityValues[index]}.png`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    }, index * 200); // Stagger downloads to avoid browser blocking
                }
            });
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

    uploadNewImage() {
        // Reset everything and show upload area
        this.currentFile = null;
        this.originalImageData = null;
        this.enhancedImageData = null;
        this.originalImagePath = null;
        this.enhancedImagePaths = null;
        
        // Hide results and enhance card
        this.resultsCard.classList.add('d-none');
        this.enhanceCard.classList.add('d-none');
        this.hideError();
        
        // Reset upload area
        this.resetUploadArea();
        
        // Clear file input
        this.imageInput.value = '';
        
        // Scroll to upload area
        this.uploadArea.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    generateReport() {
        if (!this.originalImagePath || !this.enhancedImagePaths) {
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
                original_image_path: this.originalImagePath,
                processed_images_paths: this.enhancedImagePaths,
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

    // Case History Methods
    async loadCaseHistory() {
        try {
            const response = await fetch('/api/my-cases');
            const data = await response.json();
            
            if (response.ok && data.success) {
                this.displayCaseHistory(data.cases);
            } else {
                this.showCaseHistoryError(data.error || 'Failed to load case history');
            }
        } catch (error) {
            console.error('Error loading case history:', error);
            this.showCaseHistoryError('Network error. Please check your connection.');
        }
    }

    displayCaseHistory(cases) {
        const casesList = document.getElementById('casesList');
        const casesContent = document.getElementById('casesContent');
        const casesLoading = document.getElementById('casesLoading');
        const noCasesMessage = document.getElementById('noCasesMessage');

        // Hide loading
        casesLoading.classList.add('d-none');

        if (!cases || cases.length === 0) {
            noCasesMessage.classList.remove('d-none');
            return;
        }

        // Show content
        casesContent.classList.remove('d-none');
        noCasesMessage.classList.add('d-none');

        // Generate accordion HTML
        const accordionId = 'casesAccordion';
        let accordionHTML = `<div class="accordion" id="${accordionId}">`;
        
        cases.forEach((caseData, index) => {
            const caseId = caseData.case_id;
            const accordionItemId = `case-${index}`;
            const isExpanded = index === 0; // First case expanded by default
            
            accordionHTML += `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading-${accordionItemId}">
                        <button class="accordion-button ${isExpanded ? '' : 'collapsed'}" type="button" 
                                data-bs-toggle="collapse" data-bs-target="#collapse-${accordionItemId}" 
                                aria-expanded="${isExpanded}" aria-controls="collapse-${accordionItemId}">
                            <div class="d-flex justify-content-between align-items-center w-100 me-3">
                                <div>
                                    <strong>Case ID:</strong> ${caseId}
                                </div>
                                <div class="text-muted small">
                                    <span class="badge bg-primary me-2">${caseData.total_files} files</span>
                                    <span class="badge bg-success me-2">${caseData.enhanced_files} enhanced</span>
                                    ${caseData.error_files > 0 ? `<span class="badge bg-danger">${caseData.error_files} errors</span>` : ''}
                                </div>
                            </div>
                        </button>
                    </h2>
                    <div id="collapse-${accordionItemId}" class="accordion-collapse collapse ${isExpanded ? 'show' : ''}" 
                         aria-labelledby="heading-${accordionItemId}" data-bs-parent="#${accordionId}">
                        <div class="accordion-body">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <small class="text-muted">
                                        <strong>First Upload:</strong> ${caseData.first_upload}
                                    </small>
                                </div>
                                <div class="col-md-6">
                                    <small class="text-muted">
                                        <strong>Last Upload:</strong> ${caseData.last_upload}
                                    </small>
                                </div>
                            </div>
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead class="table-light">
                                        <tr>
                                            <th>Type</th>
                                            <th>Original File</th>
                                            <th>Status</th>
                                            <th>Upload Time</th>
                                            <th>Enhancement Time</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${caseData.enhancements.map(enhancement => `
                                            <tr>
                                                <td>
                                                    <span class="badge ${enhancement.enhancement_type === 'GENAI' ? 'bg-info' : 'bg-primary'}">
                                                        ${enhancement.enhancement_type}
                                                    </span>
                                                </td>
                                                <td>
                                                    <small>${enhancement.original_filename}</small>
                                                    <br>
                                                    <small class="text-muted">${this.formatFileSize(enhancement.file_size)}</small>
                                                </td>
                                                <td>
                                                    ${enhancement.status === 'enhanced' ? 
                                                        '<span class="badge bg-success"><i class="fas fa-check"></i> Enhanced</span>' :
                                                        enhancement.status === 'error' ?
                                                        '<span class="badge bg-danger"><i class="fas fa-exclamation-triangle"></i> Error</span>' :
                                                        '<span class="badge bg-warning"><i class="fas fa-clock"></i> Uploaded</span>'
                                                    }
                                                </td>
                                                <td><small>${enhancement.upload_time}</small></td>
                                                <td><small>${enhancement.enhancement_time || '-'}</small></td>
                                                <td>
                                                    ${enhancement.can_download_report ? 
                                                        `<a class="btn btn-sm btn-outline-primary" href="/api/download-report/${enhancement.id}" target="_blank" download>
                                                            <i class="fas fa-file-pdf"></i> Download Report
                                                        </a>` :
                                                        enhancement.status === 'uploaded' ?
                                                        '<small class="text-muted">Processing...</small>' :
                                                        '<small class="text-muted">Report Not Available</small>'
                                                    }
                                                </td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        accordionHTML += '</div>';
        casesList.innerHTML = accordionHTML;
    }

    showCaseHistoryError(message) {
        const casesLoading = document.getElementById('casesLoading');
        const casesContent = document.getElementById('casesContent');
        const noCasesMessage = document.getElementById('noCasesMessage');
        
        casesLoading.classList.add('d-none');
        casesContent.classList.add('d-none');
        noCasesMessage.classList.remove('d-none');
        
        noCasesMessage.innerHTML = `
            <i class="fas fa-exclamation-triangle fa-3x text-danger mb-3"></i>
            <h5 class="text-danger">Error Loading Case History</h5>
            <p class="text-muted">${message}</p>
            <button class="btn btn-primary" onclick="app.loadCaseHistory()">
                <i class="fas fa-refresh"></i> Try Again
            </button>
        `;
    }

    formatFileSize(bytes) {
        if (!bytes) return '-';
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1048576).toFixed(1)} MB`;
    }

    async downloadCaseReport(caseId, enhancementType, originalFilename, fileId) {
        try {
            // Show loading state
            const btn = event.target.closest('button');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
            btn.disabled = true;

            if (!fileId) {
                throw new Error('File ID not found');
            }

            const response = await fetch('/api/download-case-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ file_id: fileId })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `case_${caseId}_${enhancementType}_report.pdf`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
                
                this.showNotification('Report downloaded successfully!', 'success');
            } else {
                const data = await response.json();
                throw new Error(data.error || 'Failed to generate report');
            }
        } catch (error) {
            console.error('Error downloading case report:', error);
            this.showNotification(error.message || 'Failed to download report', 'danger');
        } finally {
            // Restore button state
            const btn = event.target.closest('button');
            btn.innerHTML = '<i class="fas fa-file-pdf"></i> Download Report';
            btn.disabled = false;
        }
    }

    showNotification(message, type = 'success') {
        // Create a simple notification
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SuperResolutionApp();
});

// Add some helpful console messages
console.log('🚀 Super Resolution App loaded successfully!');
console.log('📸 Ready to enhance your images with AI magic!'); 