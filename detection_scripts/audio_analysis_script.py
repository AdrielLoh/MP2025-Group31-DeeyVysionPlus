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
import json
import shutil
import soundfile as sf


SAMPLE_RATE = 16000
DURATION = 5
MAX_TIME_STEPS = 157
N_MELS = 128
N_MFCC = 20
MODEL_PATH = "models/audio_model_v9p2ft2.keras"
SCALER_PATH = "models/audio_scaler.joblib"
SILENCE_THRESHOLD = 0.002

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def convert_to_python_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

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

def predict_audio(file_path, output_folder, unique_tag, window_duration=5, window_hop=2.5, use_cache=True):
    # ==== CHECK FOR CACHED RESULTS ====
    cache_path = os.path.join(output_folder, "cached_results.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
        return (
            cache["prediction_class"],
            cache["mel_spectrogram_path"],
            cache["mfcc_path"],
            cache["delta_path"],
            cache["f0_path"],
            cache["prediction_value"],
            cache["uploaded_audio"]
        )

    # Always convert to .wav and save in output_folder
    processed_audio_path = os.path.join(output_folder, f"audio_{unique_tag}.wav")

    if file_path.endswith(('.mp4', '.mov')):
        # Extract audio from video, then convert to .wav
        video_clip = VideoFileClip(file_path)
        audio_clip = video_clip.audio
        temp_audio_path = os.path.join(output_folder, f"temp_audio_{unique_tag}.mp3")
        audio_clip.write_audiofile(temp_audio_path, logger=None)
        audio_clip.close()
        video_clip.close()
        y, sr = librosa.load(temp_audio_path, sr=SAMPLE_RATE, mono=True)
        y = normalize_volume(y)
        sf.write(processed_audio_path, y, sr)  # Save as wav
        os.remove(temp_audio_path)  # Clean up temp audio
    else:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        y = normalize_volume(y)
        sf.write(processed_audio_path, y, sr)  # Save as wav

    actual_audio_path = processed_audio_path

    if np.max(np.abs(y)) < SILENCE_THRESHOLD or len(y) < sr:
        logging.warning("Audio too silent or too short. Skipping prediction.")
        return None, None

    window_length = int(window_duration * sr)
    hop_length = int(window_hop * sr)
    total_samples = len(y)
    pred_probs = []
    num_skipped = 0

    for start in range(0, total_samples - window_length + 1, hop_length):
        window_y = y[start : start + window_length]
        features, avg_energy, mel_db, mfcc, delta, f0, energy = extract_combined_features(window_y)
        if avg_energy < SILENCE_THRESHOLD:
            num_skipped += 1
            continue
        scaled_features = scaler.transform(features.reshape(-1, features.shape[-1]))
        input_data = scaled_features[..., np.newaxis][np.newaxis, ...]
        pred_prob = model.predict(input_data).flatten()[0]
        pred_probs.append(pred_prob)

    if not pred_probs:
        logging.warning("No valid windows found (all silent/short).")
        return None, None

    # === Aggregate result ===
    mean_prob = float(np.mean(pred_probs))
    num_fakes = int(np.sum([p >= 0.3306 for p in pred_probs]))
    num_windows = len(pred_probs)
    prediction_class = 1 if num_fakes > (num_windows // 2) else 0

    # ==== Plot features for FULL audio ====
    try:
        mel_db_full = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db_full = librosa.power_to_db(mel_db_full, ref=np.max)
        mfcc_full = librosa.feature.mfcc(S=mel_db_full, n_mfcc=N_MFCC)
        delta_full = librosa.feature.delta(mfcc_full)
        f0_full, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_full = np.nan_to_num(f0_full)
        rms_full = librosa.feature.rms(y=y)
        energy_full = np.sum(rms_full, axis=0, keepdims=True)
    except Exception as e:
        logging.error(f"Error extracting global features for plotting: {e}")
        mel_db_full, mfcc_full, delta_full, f0_full, energy_full = None, None, None, None, None

    plots = {
        f'mel_spectrogram_{unique_tag}.png': (mel_db_full, 'Mel Spectrogram', 'mel'),
        f'mfcc_{unique_tag}.png': (mfcc_full, 'MFCC', 'linear'),
        f'delta_mfcc_{unique_tag}.png': (delta_full, 'Delta MFCC', 'linear'),
        f'f0_{unique_tag}.png': (f0_full, 'Fundamental Frequency (F0)', None),
        f'energy_{unique_tag}.png': (energy_full, 'Energy', None)
    }
    relative_paths = {}
    for filename, (data, title, y_axis) in plots.items():
        plot_path = os.path.join(output_folder, filename)
        save_feature_plot(data, title, plot_path, y_axis=y_axis if y_axis else 'linear')
        relative_paths[filename] = os.path.relpath(plot_path, 'static').replace("\\", "/")

    # ==== Cache and return ====
    cache = {
        "prediction_class": int(prediction_class),
        "mel_spectrogram_path": relative_paths[f'mel_spectrogram_{unique_tag}.png'],
        "mfcc_path": relative_paths[f'mfcc_{unique_tag}.png'],
        "delta_path": relative_paths[f'delta_mfcc_{unique_tag}.png'],
        "f0_path": relative_paths[f'f0_{unique_tag}.png'],
        "prediction_value": mean_prob,
        "uploaded_audio": actual_audio_path
    }
    with open(cache_path, "w") as f:
        json.dump(convert_to_python_type(cache), f)

    return (
        prediction_class,
        relative_paths[f'mel_spectrogram_{unique_tag}.png'],
        relative_paths[f'mfcc_{unique_tag}.png'],
        relative_paths[f'delta_mfcc_{unique_tag}.png'],
        relative_paths[f'f0_{unique_tag}.png'],
        mean_prob,
        actual_audio_path
    )
