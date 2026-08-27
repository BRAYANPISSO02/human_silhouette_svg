// ==============================================================================
// Vectorizer - Frontend Logic (Vanilla JavaScript)
// ==============================================================================

// 1. DOM ELEMENT REFERENCES
// ------------------------------------------------------------------------------
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const loadingState = document.getElementById('loading-state');
const resultsSection = document.getElementById('results-section');
const originalPreview = document.getElementById('original-preview');
const svgContainer = document.getElementById('svg-container');
const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');

// Global variable to store the SVG string returned by the server
let currentSvgData = null;

// 2. UPLOAD EVENT HANDLING (CLICK AND DRAG & DROP)
// ------------------------------------------------------------------------------

// Open file dialog when clicking the primary blue button
browseBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent event bubbling to parent dropZone
    fileInput.click();
});

// Open file dialog when clicking anywhere inside the drop zone box
dropZone.addEventListener('click', () => {
    fileInput.click();
});

// Detect when a file is selected through the dialog
fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Visual hover effect when dragging a file over the drop zone
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('border-brand-blue', 'bg-blue-50/40');
    });
});

// Remove visual hover effect when dragging leaves the zone or is dropped
['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('border-brand-blue', 'bg-blue-50/40');
    });
});

// Capture the dropped file
dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
        handleFile(files[0]);
    }
});

// 3. MAIN PROCESSING AND BACKEND API COMMUNICATION
// ------------------------------------------------------------------------------
async function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Por favor, selecciona un archivo de imagen válido (JPG, PNG).');
        return;
    }

    // A) Instant preview of the original image in the browser
    const reader = new FileReader();
    reader.onload = (e) => {
        originalPreview.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // B) UI State transition: Hide dropzone and show loading spinner
    dropZone.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingState.classList.remove('hidden');

    // C) Package the image into FormData for HTTP POST transmission
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send request to FastAPI endpoint
        const response = await fetch('/api/vectorize', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Error en el servidor (${response.status})`);
        }

        const data = await response.json();
        currentSvgData = data.svg; // Store returned SVG content

        // D) Inject SVG code directly into the container
        svgContainer.innerHTML = currentSvgData;

        // Adjust SVG element attributes for proper responsive layout
        const svgElement = svgContainer.querySelector('svg');
        if (svgElement) {
            svgElement.classList.add('w-full', 'h-full', 'max-h-80');
            svgElement.style.width = '100%';
            svgElement.style.height = '100%';
        }

        // E) Reveal results section
        loadingState.classList.add('hidden');
        resultsSection.classList.remove('hidden');

    } catch (error) {
        console.error('Error vectorizing image:', error);
        alert(`Ocurrió un error al procesar la imagen: ${error.message}`);
        resetUI();
    }
}

// 4. SVG FILE DOWNLOAD AND UI RESET
// ------------------------------------------------------------------------------

// Download the SVG file
downloadBtn.addEventListener('click', () => {
    if (!currentSvgData) return;

    // Create an in-memory Blob with the SVG content
    const blob = new Blob([currentSvgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    // Trigger virtual anchor tag click to force file download
    const link = document.createElement('a');
    link.href = url;
    link.download = 'silueta_vectorial.svg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
});

// Button to reset and process another image
resetBtn.addEventListener('click', resetUI);

// Function to reset the screen to its initial state
function resetUI() {
    fileInput.value = '';
    currentSvgData = null;
    originalPreview.src = '';
    svgContainer.innerHTML = '';

    resultsSection.classList.add('hidden');
    loadingState.classList.add('hidden');
    dropZone.classList.remove('hidden');
}
