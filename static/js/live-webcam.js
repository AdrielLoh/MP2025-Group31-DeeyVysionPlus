function isMobileDevice() {
    const ua = navigator.userAgent;
    // Basic mobile detection
    const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
    // iPadOS 13+ reports as Macintosh
    const isIpad = ua.includes("Macintosh") && 'ontouchend' in document;
    return (mobileRegex.test(ua) && 'ontouchend' in document) || isIpad;
}
if (isMobileDevice()) {
    document.getElementById("webcamSection").style.display="none";
    document.getElementById("webcamUploadForm").style.display="none";
}

if (!isMobileDevice()) {
    const videoSelect = document.getElementById('videoSource');
    var webcamDetected = true;

    // Get list of video input devices
    function getCameras() {
        navigator.mediaDevices.enumerateDevices()
        .then(function(devices) {
            videoSelect.innerHTML = ''; // Clear previous
            let cameraCount = 0;
            devices.forEach(function(device) {
                if (device.kind === 'videoinput') {
                    cameraCount++;
                    let option = document.createElement('option');
                    option.value = device.deviceId;
                    option.text = device.label || `Camera ${videoSelect.length + 1}`;
                    videoSelect.appendChild(option);
                }
            });
            if (cameraCount === 0) {
                alert("No webcams detected on this device.");
                document.getElementById('startWebcamBtn').disabled = true;
                webcamDetected = false;
            }
        });
    }

    // Request permission as soon as possible
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            getCameras(); // This will show real camera labels!
            stream.getTracks().forEach(track => track.stop());
        })
        .catch(err => {
            getCameras();
            alert("No webcam permissions. Live webcam features will not work.");
            document.getElementById('startWebcamBtn').disabled = true;
            webcamDetected = false;
        });
    getCameras();
    navigator.mediaDevices.ondevicechange = getCameras; // Refresh if cams plugged/unplugged

    if (webcamDetected) {
        let mediaRecorder, recordedChunks = [], stream;

        document.getElementById('startWebcamBtn').onclick = async function() {
            const selectedDeviceId = videoSelect.value;
            stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    deviceId: { exact: selectedDeviceId },
                    frameRate: { ideal: 30, max: 30 }
                }, 
                audio: true
            });

            document.getElementById('webcam-preview').srcObject = stream;
            document.getElementById('webcam-preview').style = "";
            document.getElementById('stopWebcamBtn').disabled = false;
            document.getElementById('startWebcamBtn').disabled = true;
            document.getElementsByClassName('preview-placeholder')[0].style.display = "none";

            recordedChunks = [];
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
            mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
            mediaRecorder.start();
        };

        document.getElementById('stopWebcamBtn').onclick = function() {
            mediaRecorder.stop();
            stream.getTracks().forEach(track => track.stop());
            document.getElementById('webcam-preview').srcObject = null;
            document.getElementById('webcamSpinner').style.display = 'block';
            document.getElementById('stopWebcamBtn').disabled = true;

            mediaRecorder.onstop = function() {
                let blob = new Blob(recordedChunks, { type: 'video/webm' });
                let fileInput = document.getElementById('webcamFileInput');
                let dataTransfer = new DataTransfer();
                let file = new File([blob], 'webcam_recording.webm', { type: 'video/webm' });
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;

                document.getElementById('webcamUploadForm').style.display = 'block';
                document.getElementById('webcamSpinner').style.display = 'none';
            };
        };
    }
}