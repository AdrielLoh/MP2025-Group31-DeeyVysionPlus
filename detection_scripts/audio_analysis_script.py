import os
import librosa
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for Matplotlib
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Parameters
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 157  # Ensure this matches the training data
MODEL_PATH = "models/audio_detection_model_new.keras"
SCALER_PATH = "models/scaler.joblib"
SILENCE_THRESHOLD = 20  # Threshold in dB below reference for considering a segment as silence

# Load the saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the saved scaler
scaler = joblib.load(SCALER_PATH)

# Extract Mel spectrogram features for a segment
def extract_features(y):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]
    return mel_spectrogram

# Process and predict on an entire audio file
def predict_audio(file_path, output_folder):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    features = extract_features(y)
    # Reshape and scale the features as needed by the LSTM model
    features = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    features = features[np.newaxis, ...]  # Shape (1, time_steps, features)
    
    if len(features.shape) < 4:
        features = features[..., np.newaxis]  # Add a channel dimension if your model expects it
    
    prediction = model.predict(features)
    prediction_class = int(prediction[0, 0] > 0.5)
    
    # Save Mel spectrogram
    mel_spectrogram = extract_features(y)
    mel_spectrogram_path = os.path.join(output_folder, 'mel_spectrogram.png')
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(mel_spectrogram_path)
    plt.close()  # Close the figure to prevent memory leaks
    
    # Return the path relative to the static folder
    relative_path = os.path.relpath(mel_spectrogram_path, 'static').replace("\\", "/")
    return prediction_class, relative_path

def predict_real_time_audio(output_folder):
    import sounddevice as sd
    from scipy.io.wavfile import write
    
    # Define the path for saving the audio file
    audio_output_folder = 'static/uploads'  # Ensure this is the directory where the audio should be saved

    # Record audio
    duration = 5  # seconds
    logging.debug("Recording audio for real-time analysis...")
    recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    logging.debug("Recording finished.")
    
    # Save recording to a file
    audio_path = os.path.join(audio_output_folder, 'real_time_audio.wav')
    write(audio_path, SAMPLE_RATE, recording)
    
    # Perform prediction on the recorded audio
    return predict_audio(audio_path, output_folder)