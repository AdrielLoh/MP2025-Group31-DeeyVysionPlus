document.addEventListener("DOMContentLoaded", () => {
    chrome.storage.local.get("deepfakeResults", ({ deepfakeResults }) => {
        if (!deepfakeResults) {
            document.getElementById("resultContent").innerHTML =
                "<div>No results found. Please run a detection first.</div>";
            return;
        }
        const { data, methods } = deepfakeResults;
        let html = "";
        // For multiple methods, data is an object, otherwise just single result.
        if (methods.length === 1) {
            html = renderResultCard(methods[0], data);
        } else {
            for (const method of Object.keys(data)) {
                html += renderResultCard(method, data[method]);
            }
        }
        document.getElementById("resultContent").innerHTML = html;
    });
    // Simple navigation
    function goBack() {
        chrome.storage.local.remove("deepfakeResults", () => {
            window.location.href = "popup.html";
        });
    }
    const backBtn = document.getElementById("backBtn");
    if (backBtn) {
        backBtn.addEventListener("click", goBack);
    }
});

function renderResultCard(method, result) {
    let color,
        icon,
        label,
        details = "";
    switch (method) {
        case "deep_learning":
            label = "Deep Learning";
            color = "linear-gradient(90deg,#667eea,#764ba2)";
            icon = "🧠";
            if (
                result.real_count !== undefined &&
                result.fake_count !== undefined
            ) {
                details = `<div>Real: <b>${result.real_count}</b> | Fake: <b>${result.fake_count}</b></div>`;
            }
            break;
        case "physiological":
            label = "Physiological";
            color = "linear-gradient(90deg,#4facfe,#00f2fe)";
            icon = "❤️";
            break;
        case "visual_artifacts":
            label = "Visual Artifacts";
            color = "linear-gradient(90deg,#f093fb,#f5576c)";
            icon = "🔬";
            break;
        case "body_posture":
            label = "Body Posture";
            color = "linear-gradient(90deg,#fa709a,#fee140)";
            icon = "🤖";
            break;
        case "audio":
            label = "Audio Analysis";
            color = "linear-gradient(90deg,#ff9a9e,#fecfef)";
            icon = "🔊";
            if (result.prediction_value !== undefined) {
                details = `<div>Deepfake Probability: <b>${result.prediction_value}%</b></div>`;
            }
            break;
        default:
            label = method;
            color = "linear-gradient(90deg,#cecece,#888)";
            icon = "🔊";
    }
    let prediction = (result.prediction || "No result")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    let confidence =
        result.confidence !== undefined
            ? `<div>Confidence: <b>${result.confidence}%</b></div>`
            : "";

    // Audio playback section
    let audioPlayer = "";
    if (method === "audio" && result.uploaded_audio) {
        audioPlayer = `
        <div style="margin-top:12px;padding:12px;background:rgba(0,0,0,0.2);border-radius:8px;">
            <div style="font-weight:600;margin-bottom:8px;color:#fff;">🎵 Audio Sample</div>
            <audio controls style="width:100%;height:40px;border-radius:6px;background:rgba(0,0,0,0.3);">
                <source src="http://localhost:5000/${result.uploaded_audio}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        </div>`;
    }

    // Face results table with confidence & time
    let faces = "";
    if (method == "body_posture") {
        if (result.persons && Array.isArray(result.persons)) {
            faces = `
            <details style="margin-top:10px;">
            <summary style="cursor:pointer;">Person-level Results</summary>
            <table style="width:98%;margin:10px auto;font-size:.96em;border-collapse:collapse;">
                <tr>
                    <th style="text-align:left;padding:4px;">Person</th>
                    <th style="text-align:left;padding:4px;">Result</th>
                    <th style="text-align:left;padding:4px;">Confidence</th>
                    <th style="text-align:left;padding:4px;">Track Length</th>
                    <th style="text-align:left;padding:4px;">Visualizations</th>
                </tr>
                ${result.persons.map(person => `
                <tr>
                    <td style="padding:4px;">${"#" + (person.person_id ?? "")}</td>
                    <td style="padding:4px;">${person.result ?? ""}</td>
                    <td style="padding:4px;">${person.result_confidence !== undefined ? (person.result_confidence + "%") : ""}</td>
                    <td style="padding:4px;">${person.track_length ?? ""}</td>
                    <td style="padding:4px;">
                        ${(person.figure_location && person.figure_location.length)
                            ? person.figure_location.map(img =>
                                `<img src="http://localhost:5000/${img}" style="width:60px;max-height:48px;border-radius:6px;margin:2px;">`
                            ).join("")
                            : ""}
                    </td>
                </tr>
                `).join("")}
            </table>
            </details>
            `;
        }
        let video = "";
        if (result.output_video) {
            let videoPath = result.output_video;
            video = `
            <video src="http://localhost:5000/${videoPath}" controls style="width:97%;margin-top:11px;border-radius:13px;box-shadow:0 2px 12px #00f2fe2a;"></video>
            `;
        }
        return `
            <div style="border-radius: 1.1em; padding: 1em 1em 0.7em 1em; margin: 0 0 1.1em 0; color: #fff; box-shadow: var(--shadow-glow, 0 3px 12px rgba(120,119,198,0.19)); position: relative;">
                <div style="font-size:1.2em;font-weight:700;margin-bottom:0.2em;">${icon} ${label}</div>
                <div style="font-size:1.13em; margin-bottom:0.35em;"><b>Overall Prediction:</b> ${prediction}</div>
                ${confidence}
                ${details}
                ${audioPlayer}
                ${faces}
                ${video}
            </div>
            `;
    } else {
        if (result.face_results && Array.isArray(result.face_results)) {
            faces = `
            <details style="margin-top:10px;">
            <summary style="cursor:pointer;">Face-level Results</summary>
            <table style="width:98%;margin:10px auto;font-size:.96em;border-collapse:collapse;">
                <tr>
                <th style="text-align:left;padding:4px;">Face</th>
                <th style="text-align:left;padding:4px;">Result</th>
                <th style="text-align:left;padding:4px;">Confidence</th>
                </tr>
                ${result.face_results.map(face => `
                <tr>
                    <td style="padding:4px;">${"#" + face.track_id ?? ""}</td>
                    <td style="padding:4px;">${face.result ?? ""}</td>
                    <td style="padding:4px;">${face.confidence !== undefined ? (face.confidence + "%") : ""}</td>
                </tr>
                `).join("")}
            </table>
            </details>
            `;
        }
        // Output video download/display
        let video = "";
        if (result.output_video) {
            let videoPath = result.output_video;
            video = `
            <video src="http://localhost:5000/${videoPath}" controls style="width:97%;margin-top:11px;border-radius:13px;box-shadow:0 2px 12px #00f2fe2a;"></video>
            `;
        }

        // Plots/images
        let plots = "";
        if (result.rvf_plot)    // Can replace localhost with actual domain if our app is hosted 
            plots += `<img src="http://localhost:5000/${result.rvf_plot}" style="width:90%;margin-top:7px;border-radius:14px;box-shadow:0 2px 12px #667eea2a;">`;
        if (result.conf_plot)
            plots += `<img src="http://localhost:5000/${result.conf_plot}" style="width:90%;margin-top:7px;border-radius:14px;box-shadow:0 2px 12px #764ba22a;">`;
        if (result.mel_spectrogram_path)
            plots += `<img src="http://localhost:5000/static/${result.mel_spectrogram_path}" style="width:90%;margin-top:7px;border-radius:14px;box-shadow:0 2px 12px #764ba22a;">`;
        if (result.mfcc_path)
            plots += `<img src="http://localhost:5000/static/${result.mfcc_path}" style="width:90%;margin-top:7px;border-radius:14px;box-shadow:0 2px 12px #764ba22a;">`;
        
        return `
            <div style=" 
                border-radius: 1.1em; 
                padding: 1em 1em 0.7em 1em; 
                margin: 0 0 1.1em 0;
                color: #fff;
                box-shadow: var(--shadow-glow, 0 3px 12px rgba(120,119,198,0.19));
                position: relative;
                ">
                <div style="font-size:1.2em;font-weight:700;margin-bottom:0.2em;">${icon} ${label}</div>
                <div style="font-size:1.13em; margin-bottom:0.35em;"><b>Prediction:</b> ${prediction}</div>
                ${confidence}
                ${details}
                ${audioPlayer}
                ${faces}
                ${video}
                ${plots}
            </div>`;
    }
}
