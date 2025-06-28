const videoSelect = document.getElementById('videoSource');
// Get list of video input devices
function getCameras() {
    navigator.mediaDevices.enumerateDevices()
    .then(function(devices) {
        videoSelect.innerHTML = ''; // Clear previous
        devices.forEach(function(device) {
            if (device.kind === 'videoinput') {
                let option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${videoSelect.length + 1}`;
                videoSelect.appendChild(option);
            }
        });
    });
}
getCameras();
navigator.mediaDevices.ondevicechange = getCameras; // Refresh if cams plugged/unplugged

let mediaRecorder, recordedChunks = [], stream;

document.getElementById('startWebcamBtn').onclick = async function() {
    const selectedDeviceId = videoSelect.value;
    stream = await navigator.mediaDevices.getUserMedia({
        video: { 
            deviceId: { exact: selectedDeviceId },
            frameRate: { ideal: 30, max: 30 }
        }, 
        audio: false
    });

    document.getElementById('webcam-preview').srcObject = stream;
    document.getElementById('webcam-preview').style = "";
    document.getElementById('stopWebcamBtn').disabled = false;
    document.getElementById('startWebcamBtn').disabled = true;

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