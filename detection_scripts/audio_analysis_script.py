# --- Updated Production Script using New Feature Extraction and Model ---

import os
import librosa
import numpy as np
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

SAMPLE_RATE = 16000
DURATION = 5
MAX_TIME_STEPS = 157
N_MELS = 128
N_MFCC = 20
MODEL_PATH = "models/audio_model_tf217_alcj_v2.keras"
SCALER_PATH = "models/audio_scaler.joblib"
SILENCE_THRESHOLD = 0.002

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def pad_or_truncate(feature, max_len):
    if feature.shape[1] < max_len:
        return np.pad(feature, ((0, 0), (0, max_len - feature.shape[1])), mode='constant')
    else:
        return feature[:, :max_len]

def normalize_volume(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def extract_combined_features(y, sr=SAMPLE_RATE):
    features = []
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    features += [pad_or_truncate(mel_db, MAX_TIME_STEPS), pad_or_truncate(mfcc, MAX_TIME_STEPS),
                 pad_or_truncate(delta, MAX_TIME_STEPS)]
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    features.append(pad_or_truncate(np.nan_to_num(f0)[np.newaxis, :], MAX_TIME_STEPS))
    energy = np.sum(librosa.feature.rms(y=y), axis=0, keepdims=True)
    features.append(pad_or_truncate(energy, MAX_TIME_STEPS))
    return np.vstack(features), np.mean(energy)

def predict_audio(file_path, output_folder):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    y = normalize_volume(y)
    features, avg_energy = extract_combined_features(y)

    if avg_energy < SILENCE_THRESHOLD:
        logging.warning("Audio too silent. Skipping prediction.")
        return None, None

    features = scaler.transform(features.reshape(-1, features.shape[-1]))
    input_data = features[..., np.newaxis][np.newaxis, ...]
    pred_prob = model.predict(input_data).flatten()[0]
    prediction_class = 1 if pred_prob >= 0.3368 else 0

    mel = features[:N_MELS].reshape(N_MELS, MAX_TIME_STEPS)
    mel_spectrogram_path = os.path.join(output_folder, 'mel_spectrogram.png')
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(mel_spectrogram_path)
    plt.close()

    relative_path = os.path.relpath(mel_spectrogram_path, 'static').replace("\\", "/")
    return prediction_class, relative_path

def predict_real_time_audio(output_folder):
    import sounddevice as sd
    from scipy.io.wavfile import write
    audio_output_folder = 'static/uploads'

    logging.debug("Recording audio for real-time analysis...")
    recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    logging.debug("Recording finished.")

    audio_path = os.path.join(audio_output_folder, 'real_time_audio.wav')
    write(audio_path, SAMPLE_RATE, recording)
    return predict_audio(audio_path, output_folder)
