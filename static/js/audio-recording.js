// Audio source selection
const audioSelect = document.getElementById('audioSource');

function getAudioSources() {
    navigator.mediaDevices.enumerateDevices()
    .then(function(devices) {
        audioSelect.innerHTML = '';
        devices.forEach(function(device) {
            if (device.kind === 'audioinput') {
                let option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Microphone ${audioSelect.length + 1}`;
                audioSelect.appendChild(option);
            }
        });
    });
}
// Request permission as soon as possible
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        getAudioSources(); // This will show real camera labels!
        stream.getTracks().forEach(track => track.stop());
    })
    .catch(err => {
        getAudioSources();
        alert("Microphone access denied. Recording features will not work.");
    });
getAudioSources();
navigator.mediaDevices.ondevicechange = getAudioSources;

// Live audio recording functionality
let mediaRecorder, recordedChunks = [], audioStream, audioContext, analyser, startTime;
const volumeBars = document.querySelectorAll('.volume-bar');
const volumeLevel = document.getElementById('volume-level');
const recordingTimer = document.getElementById('recording-timer');
let animationId;

function updateVolumeVisualization() {
    if (!analyser) return;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    
    // Calculate average volume
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
    }
    const average = sum / bufferLength;
    const volumePercent = Math.round((average / 255) * 100);
    
    // Update volume level text
    volumeLevel.textContent = `Volume: ${volumePercent}%`;
    
    // Update volume bars
    const activeBarCount = Math.floor((average / 255) * volumeBars.length);
    volumeBars.forEach((bar, index) => {
        if (index < activeBarCount) {
            bar.classList.add('active');
        } else {
            bar.classList.remove('active');
        }
    });
    
    animationId = requestAnimationFrame(updateVolumeVisualization);
}

function updateTimer() {
    if (!startTime) return;
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    recordingTimer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

document.getElementById('startAudioBtn').onclick = async function() {
    try {
        const selectedDeviceId = audioSelect.value;
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: { 
                deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });

        // Set up audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);
        analyser.fftSize = 256;

        // Show volume meter and hide placeholder
        document.getElementById('audio-preview-placeholder').style.display = 'none';
        document.getElementById('volume-meter').style.display = 'block';
        
        // Start visualization
        updateVolumeVisualization();
        
        // Start recording
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(audioStream);
        mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
        mediaRecorder.start();
        
        // Update UI
        document.getElementById('startAudioBtn').disabled = true;
        document.getElementById('stopAudioBtn').disabled = false;
        
        // Start timer
        startTime = Date.now();
        recordingTimer.style.display = 'block';
        setInterval(updateTimer, 1000);
        
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Unable to access microphone. Please check permissions.');
    }
};

document.getElementById('stopAudioBtn').onclick = function() {
    mediaRecorder.stop();
    audioStream.getTracks().forEach(track => track.stop());
    
    // Stop visualization
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    if (audioContext) {
        audioContext.close();
    }
    
    // Show spinner
    document.getElementById('audioSpinner').style.display = 'block';
    document.getElementById('stopAudioBtn').disabled = true;
    
    // Reset timer
    startTime = null;
    recordingTimer.style.display = 'none';

    mediaRecorder.onstop = async function() {
        try {
            // Create blob from recorded chunks
            const webmBlob = new Blob(recordedChunks, { type: 'audio/webm' });
            
            // Convert WebM to WAV
            const wavBlob = await convertToWav(webmBlob);
            
            // Generate random filename
            const randomString = Math.random().toString(36).substring(2, 15);
            const filename = `recorded_audio${randomString}.wav`;
            
            // Create file and add to form
            const fileInput = document.getElementById('audioFileInput');
            const dataTransfer = new DataTransfer();
            const file = new File([wavBlob], filename, { type: 'audio/wav' });
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;

            document.getElementById('audioUploadForm').style.display = 'block';
            document.getElementById('audioSpinner').style.display = 'none';
            
            // Reset UI
            document.getElementById('audio-preview-placeholder').style.display = 'block';
            document.getElementById('volume-meter').style.display = 'none';
            document.getElementById('startAudioBtn').disabled = false;
            
            // Reset volume bars
            volumeBars.forEach(bar => bar.classList.remove('active'));
            volumeLevel.textContent = 'Volume: 0%';
            
        } catch (error) {
            console.error('Error processing audio:', error);
            alert('Error processing audio recording. Please try again.');
            
            // Reset UI on error
            document.getElementById('audioSpinner').style.display = 'none';
            document.getElementById('audio-preview-placeholder').style.display = 'block';
            document.getElementById('volume-meter').style.display = 'none';
            document.getElementById('startAudioBtn').disabled = false;
            document.getElementById('stopAudioBtn').disabled = true;
        }
    };
};

// Function to convert WebM blob to WAV
async function convertToWav(webmBlob) {
    return new Promise((resolve, reject) => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const fileReader = new FileReader();
        
        fileReader.onload = function(e) {
            audioContext.decodeAudioData(e.target.result)
                .then(audioBuffer => {
                    const wavBlob = audioBufferToWav(audioBuffer);
                    resolve(wavBlob);
                })
                .catch(error => {
                    console.error('Error decoding audio:', error);
                    // If conversion fails, fallback to original blob but change extension
                    const randomString = Math.random().toString(36).substring(2, 15);
                    resolve(new Blob([webmBlob], { type: 'audio/webm' }));
                });
        };
        
        fileReader.onerror = () => reject(new Error('Failed to read audio file'));
        fileReader.readAsArrayBuffer(webmBlob);
    });
}

// Function to convert AudioBuffer to WAV blob
function audioBufferToWav(buffer) {
    const length = buffer.length;
    const numberOfChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numberOfChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = length * blockAlign;
    const bufferSize = 44 + dataSize;
    
    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);
    
    // Convert audio data
    let offset = 44;
    for (let i = 0; i < length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

// Handle audio upload form submission
document.getElementById('audioUploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Show processing state
    const submitBtn = this.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';
    
    // Submit form
    this.submit();
});