// Global state
let currentImageBase64 = null;
let currentAnalysis = null;
let currentFile = null; // Store the uploaded file for analysis

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    checkApiStatus();
    setupFileUpload();
    setupAnalyzeButton();
    setupDownloadButton();
});

// Check API status
async function checkApiStatus() {
    const statusBox = document.getElementById('api-status');
    const statusText = document.getElementById('status-text');
    
    if (!statusBox || !statusText) {
        console.error('Status elements not found');
        return;
    }
    
    try {
        const response = await fetch('/api/health');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        if (data.api_configured) {
            statusBox.className = 'status-box success';
            statusText.textContent = 'Workspace ready!';
        } else {
            statusBox.className = 'status-box error';
            statusText.textContent = 'Google API key is not configured on the server. Please set GOOGLE_API_KEY in your environment or .env file.';
        }
    } catch (error) {
        console.error('Error checking API status:', error);
        statusBox.className = 'status-box error';
        statusText.textContent = 'Error checking API status: ' + error.message;
    }
}

// Setup file upload
function setupFileUpload() {
    const fileInput = document.getElementById('file-input');
    const uploadLabel = document.querySelector('.upload-label');
    
    // Drag and drop handlers
    const uploadContainer = document.querySelector('.upload-container');
    
    uploadContainer.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadContainer.style.borderColor = 'var(--primary-color)';
        uploadContainer.style.backgroundColor = '#f0f8ff';
    });
    
    uploadContainer.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadContainer.style.borderColor = 'var(--border-color)';
        uploadContainer.style.backgroundColor = 'var(--card-bg)';
    });
    
    uploadContainer.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadContainer.style.borderColor = 'var(--border-color)';
        uploadContainer.style.backgroundColor = 'var(--card-bg)';

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input change handler
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// Handle file selection
async function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/dicom', 'application/pdf', 'application/octet-stream'];
    const validExtensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom', '.pdf'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        showError('Invalid file type. Please upload JPG, JPEG, PNG, DICOM, or PDF files.');
        return;
    }
    
    // Show preview
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('preview-image').src = data.preview;
            currentImageBase64 = data.preview;
            currentFile = file; // Store the file for later analysis
            document.getElementById('image-section').classList.remove('hidden');
            document.getElementById('analysis-section').classList.add('hidden');
            hideError();
        } else {
            showError(data.error || 'Failed to upload image.');
        }
    } catch (error) {
        showError('Error uploading image: ' + error.message);
    }
}

// Setup analyze button
function setupAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.addEventListener('click', async function() {
        // Use stored file if available, otherwise try file input
        const file = currentFile || (document.getElementById('file-input').files.length > 0 ? document.getElementById('file-input').files[0] : null);
        
        if (!file) {
            showError('Please upload an image or PDF first.');
            return;
        }
        
        await analyzeImage(file);
    });
}

// Analyze image
async function analyzeImage(file) {
    const loadingDiv = document.getElementById('loading');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analysisSection = document.getElementById('analysis-section');
    
    // Show loading, hide results, disable button
    loadingDiv.classList.remove('hidden');
    analysisSection.classList.add('hidden');
    analyzeBtn.disabled = true;
    hideError();
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Handle PDF (page-wise) or single image
            if (data.is_pdf && data.page_results) {
                // PDF with multiple pages
                currentAnalysis = data.page_results;
                currentImageBase64 = data.page_results[0].image_base64; // Use first page for download
                displayPDFAnalysis(data.page_results);
            } else {
                // Single image
                currentAnalysis = data.analysis;
                currentImageBase64 = data.image_base64;
                displayAnalysis(data.analysis);
            }
            analysisSection.classList.remove('hidden');
        } else {
            showError(data.error || 'Analysis failed.');
        }
    } catch (error) {
        showError('Error analyzing image: ' + error.message);
    } finally {
        loadingDiv.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

// Display PDF analysis (page-wise)
function displayPDFAnalysis(pageResults) {
    const analysisContent = document.getElementById('analysis-content');
    let html = '';
    
    pageResults.forEach((page, index) => {
        html += `<div class="pdf-page-analysis" style="margin-bottom: 2rem; padding-bottom: 2rem; border-bottom: 2px solid var(--border-color);">`;
        html += `<h2 style="color: var(--primary-color); margin-bottom: 1rem;">Page ${page.page_number}</h2>`;
        html += convertMarkdownToHTML(page.analysis);
        html += `</div>`;
    });
    
    analysisContent.innerHTML = html;
}

// Convert markdown to HTML
function convertMarkdownToHTML(markdown) {
    let html = markdown;
    
    // Convert headers (order matters - do ### before ## before #)
    html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    // Convert bold text
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
    
    // Convert italic text
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/_(.*?)_/g, '<em>$1</em>');
    
    // Convert lists (handle both - and *)
    html = html.replace(/^[-*] (.*$)/gm, '<li>$1</li>');
    
    // Wrap consecutive list items in <ul> tags
    html = html.replace(/(<li>.*<\/li>(\n|$))+/g, function(match) {
        return '<ul>' + match.replace(/\n/g, '') + '</ul>';
    });
    
    // Convert horizontal rules
    html = html.replace(/^---$/gm, '<hr>');
    
    // Split into paragraphs (double newlines)
    let paragraphs = html.split(/\n\n+/);
    html = paragraphs.map(p => {
        p = p.trim();
        if (!p) return '';
        // Don't wrap if it's already a block element
        if (p.match(/^<(h[1-6]|ul|ol|hr)/)) {
            return p;
        }
        return '<p>' + p + '</p>';
    }).join('\n');
    
    // Clean up any remaining single newlines within paragraphs
    html = html.replace(/(<p>.*?<\/p>)/gs, function(match) {
        return match.replace(/\n/g, ' ').replace(/\s+/g, ' ');
    });
    
    return html;
}

// Display analysis results
function displayAnalysis(analysis) {
    const analysisContent = document.getElementById('analysis-content');
    
    // Convert markdown to HTML (improved conversion)
    analysisContent.innerHTML = convertMarkdownToHTML(analysis);
}

// Setup download button
function setupDownloadButton() {
    const downloadBtn = document.getElementById('download-btn');
    downloadBtn.addEventListener('click', async function() {
        if (!currentAnalysis || !currentImageBase64) {
            showError('No analysis available to download.');
            return;
        }
        
        await downloadDocx();
    });
}

// Download DOCX
async function downloadDocx() {
    try {
        let requestBody;
        
        // Handle PDF (multiple pages) or single image
        if (Array.isArray(currentAnalysis)) {
            // PDF: Send page results with images
            requestBody = {
                page_results: currentAnalysis
            };
        } else {
            // Single image: Send analysis and image
            requestBody = {
                analysis: currentAnalysis,
                image_base64: currentImageBase64
            };
        }
        
        const response = await fetch('/api/download-docx', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'medical_image_analysis.docx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const data = await response.json();
            showError(data.error || 'Failed to download document.');
        }
    } catch (error) {
        showError('Error downloading document: ' + error.message);
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide error message
function hideError() {
    const errorDiv = document.getElementById('error-message');
    errorDiv.classList.add('hidden');
}

