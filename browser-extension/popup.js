const API_BASE = "http://localhost:5000";

function getSelectedMethods() {
    return Array.from(document.querySelectorAll(".method:checked")).map(
        (x) => x.value
    );
}

document.getElementById("detectBtn").onclick = async () => {
    const status = document.getElementById("status");
    const resultBox = document.getElementById("result");
    status.textContent = "";
    resultBox.innerHTML = "";
    const detectBtn = document.getElementById("detectBtn")

    const fileInput = document.getElementById("fileInput");
    const urlInput = document.getElementById("videoUrl");
    const videoUrl = urlInput.value.trim();
    const methods = getSelectedMethods();

    if (!fileInput.files.length && !videoUrl) {
        status.textContent = "Select a file or paste a video URL!";
        return;
    }
    if (methods.length === 0) {
        status.textContent = "Please select at least one method.";
        return;
    }
    status.textContent = "Uploading & analyzing...";
    detectBtn.disabled = true;
    let formData = new FormData();
    let endpoint = "";

    if (methods.length === 1) {
        if (methods[0] === "deep_learning") {
            endpoint = "/deep_learning_static";
        } else if (methods[0] === "physiological") {
            endpoint = "/physiological_signal_try";
        } else if (methods[0] === "visual_artifacts") {
            endpoint = "/visual_artifacts_static";
        } else if (methods[0] === "body_posture") {
            endpoint = "/body_posture_detect";
        } else if (methods[0] === "audio") {
            endpoint = "/audio_analysis";
        }
        if (fileInput.files.length) formData.append("file", fileInput.files[0]);
        if (videoUrl) formData.append("video_url", videoUrl);
        if (methods[0] === "physiological")
            formData.append("detection_method", "machine");
    } else {
        endpoint = "/multi_detection";
        if (fileInput.files.length) formData.append("file", fileInput.files[0]);
        if (videoUrl) formData.append("video_url", videoUrl);
        for (let m of methods) formData.append("methods", m);
    }

    try {
        const response = await fetch(API_BASE + endpoint, {
            method: "POST",
            body: formData,
            headers: { Accept: "application/json" },
        });

        const contentType = response.headers.get("content-type");
        const rawText = await response.text(); // Only read ONCE

        let data;
        try {
            if (contentType && contentType.includes("application/json")) {
                data = JSON.parse(rawText);
            } else {
                data = null; // Not JSON
            }
        } catch (e) {
            data = null; // Not JSON
        }

        status.textContent = "Analysis complete!";
        if (data) {
            // Save results and redirect
            chrome.storage.local.set(
                { deepfakeResults: { data, methods } },
                () => {
                    window.location.href = "results.html";
                }
            );
        } else {
            // If not JSON: fallback to HTML
            resultBox.innerHTML = rawText;
        }
    } catch (e) {
        status.textContent = "Error: " + e.message;
    }
};
