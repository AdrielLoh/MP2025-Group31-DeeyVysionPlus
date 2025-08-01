document.addEventListener("DOMContentLoaded", function() {
    let fileInput = document.getElementById("file");
    let dropbox = document.getElementById("dropbox");
    let videoPreview = document.getElementById("video-preview");
    let uploadContent = document.querySelector(".upload-content");
    let uploadIcon = document.querySelector(".upload-icon");

    // Handle file selection
    fileInput.addEventListener("change", function(event) {
        handleFileUpload(event.target.files[0]);
    });

    // Handle drag and drop
    dropbox.addEventListener("dragover", function(event) {
        event.preventDefault();
        dropbox.classList.add("dragover");
    });

    dropbox.addEventListener("dragleave", function(event) {
        event.preventDefault();
        dropbox.classList.remove("dragover");
    });

    dropbox.addEventListener("drop", function(event) {
        event.preventDefault();
        dropbox.classList.remove("dragover");
        let files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileUpload(files[0]);
        }
    });

    function handleFileUpload(file) {
        if (file && file.type.startsWith('video/')) {
            // Clear any existing file info and success messages first
            clearPreviousUpload();
            
            // Show upload success state
            dropbox.classList.add("file-uploaded");
            
            // Hide upload content and show preview
            uploadContent.style.display = "none";
            uploadIcon.style.display = "none";
            
            // Create and show video preview
            let videoURL = URL.createObjectURL(file);
            videoPreview.src = videoURL;
            videoPreview.style.display = "block";
            
            // Show file info
            showFileInfo(file);
            
            // Enable submit button
            submitButton.disabled = false;
            submitButton.classList.add("ready");
            
            // Show success message
            showUploadSuccess(file.name);
        } else {
            alert("Please select a valid video file.");
        }
    }

    function clearPreviousUpload() {
        // Remove any existing file info and success messages
        let existingFileInfo = document.querySelectorAll(".file-info");
        let existingSuccessMessage = document.querySelectorAll(".upload-success");
        
        existingFileInfo.forEach(element => element.remove());
        existingSuccessMessage.forEach(element => element.remove());
        
        // Clear video preview
        if (videoPreview.src) {
            URL.revokeObjectURL(videoPreview.src);
        }
    }

    function showFileInfo(file) {
        let fileInfo = document.createElement("div");
        fileInfo.className = "file-info";
        fileInfo.innerHTML = `
            <div class="file-details">
                <div class="file-icon">🎬</div>
                <div class="file-text">
                    <h4>${file.name}</h4>
                    <p>Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
                <button type="button" class="remove-file" onclick="removeFile()">×</button>
            </div>
        `;
        
        // Insert before video preview
        dropbox.insertBefore(fileInfo, videoPreview);
    }   

    // Make removeFile function global
    window.removeFile = function() {
        // Clear any previous uploads first
        clearPreviousUpload();
        
        // Reset dropbox to original state
        dropbox.classList.remove("file-uploaded");
        uploadContent.style.display = "block";
        uploadIcon.style.display = "block";
        videoPreview.style.display = "none";
        videoPreview.src = "";
        
        // Reset file input
        fileInput.value = "";
        
        // Disable submit button
        submitButton.disabled = true;
        submitButton.classList.remove("ready");
    };
});
document.addEventListener("DOMContentLoaded", function() {
    let dvurlForm = document.getElementById("dvurl-form");
    let dvurlLoading = document.getElementById("dvurl-loading");
    let submitButton = document.getElementById("submitButton");
    let dvurlSubmit = document.getElementById("dvurl-submit");

    if (dvurlForm) {
        dvurlForm.addEventListener("submit", function(event) {
            dvurlSubmit.disabled = true;
            submitButton.disabled = true;
            dvurlSubmit.innerHTML = "Processing...";
            dvurlLoading.style.display = "block";
        });
    }
});
document.addEventListener("DOMContentLoaded", function() {
    let webcamUploadForm = document.getElementById("webcamUploadForm");
    let webcamSubmitBtn = document.getElementById("webcamSubmit");
    let dvurlSubmit = document.getElementById("dvurl-submit");
    let submitButton = document.getElementById("submitButton");

    webcamUploadForm.addEventListener("submit", function(event) {
        dvurlSubmit.disabled = true;
        submitButton.disabled = true;
        webcamSubmitBtn.disabled = true;
        webcamSubmitBtn.innerHTML = "Processing...";
    });
});