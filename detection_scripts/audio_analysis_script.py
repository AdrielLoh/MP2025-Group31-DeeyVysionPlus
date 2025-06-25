# --- Updated Production Script with Additional Feature Visualizations ---
# To do improvements:
# Batch processing and prediction
# Better error handling

import os
import librosa
import numpy as np
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip

SAMPLE_RATE = 16000
DURATION = 5
MAX_TIME_STEPS = 157
N_MELS = 128
N_MFCC = 20
MODEL_PATH = "models/audio_model_v9p2ft2.keras" # v8 is still the best
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
    f0 = np.nan_to_num(f0)
    features.append(pad_or_truncate(f0[np.newaxis, :], MAX_TIME_STEPS))
    rms = librosa.feature.rms(y=y)
    energy = np.sum(rms, axis=0, keepdims=True)
    avg_energy = np.mean(energy) if energy.size > 0 and np.max(energy) > 0 else 0.0
    features.append(pad_or_truncate(energy, MAX_TIME_STEPS))
    return np.vstack(features), avg_energy, mel_db, mfcc, delta, f0, energy

def save_feature_plot(feature, title, file_path, y_axis='linear'):
    plt.figure(figsize=(10, 4))
    try:
        if feature.ndim == 2:
            librosa.display.specshow(feature, sr=SAMPLE_RATE, x_axis='time', y_axis=y_axis)
            plt.colorbar(format='%+2.0f dB')
        else:
            plt.plot(feature.flatten())
        plt.title(title)
        plt.tight_layout()
        plt.savefig(file_path)
    except Exception as e:
        logging.error(f"Failed to save plot for {title}: {e}")
    finally:
        plt.close()

def predict_audio(file_path, output_folder):
    if file_path.endswith('.wav') or file_path.endswith('.mp3') or file_path.endswith('.flac') or file_path.endswith('.opus'):
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        y = normalize_volume(y)
        features, avg_energy, mel_db, mfcc, delta, f0, energy = extract_combined_features(y)
    elif file_path.endswith('.mp4') or file_path.endswith('.mov'):
        video_clip = VideoFileClip(file_path)
        audio_clip = video_clip.audio
        saved_audio = "static/uploads/separated_audio.mp3"
        audio_clip.write_audiofile(saved_audio)
        audio_clip.close()
        video_clip.close()
        y, sr = librosa.load(saved_audio, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        y = normalize_volume(y)
        features, avg_energy, mel_db, mfcc, delta, f0, energy = extract_combined_features(y)
    else:
        return None, None

    if avg_energy < SILENCE_THRESHOLD:
        logging.warning("Audio too silent. Skipping prediction.")
        return None, None

    features = scaler.transform(features.reshape(-1, features.shape[-1]))
    input_data = features[..., np.newaxis][np.newaxis, ...]
    pred_prob = model.predict(input_data).flatten()[0]
    prediction_class = 1 if pred_prob >= 0.3306 else 0
    # 8p2=0.3368, 9p1=0.2798, 9p2=0.3798, 9p2ft1=0.3656, 9p2ft2=0.3306

    # Save visualizations
    plots = {
        'mel_spectrogram.png': (mel_db, 'Mel Spectrogram', 'mel'),
        'mfcc.png': (mfcc, 'MFCC', 'linear'),
        'delta_mfcc.png': (delta, 'Delta MFCC', 'linear'),
        'f0.png': (f0, 'Fundamental Frequency (F0)', None),
        'energy.png': (energy, 'Energy', None)
    }

    relative_paths = {}
    for filename, (data, title, y_axis) in plots.items():
        plot_path = os.path.join(output_folder, filename)
        save_feature_plot(data, title, plot_path, y_axis=y_axis if y_axis else 'linear')
        relative_paths[filename] = os.path.relpath(plot_path, 'static').replace("\\", "/")

    return prediction_class, relative_paths['mel_spectrogram.png'], relative_paths['mfcc.png'], relative_paths['delta_mfcc.png'], relative_paths['f0.png'], pred_prob

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
