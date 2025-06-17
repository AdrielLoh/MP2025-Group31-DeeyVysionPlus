# --- Script 4: Testing the model with predicting a single file (last step to deployment) ---

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import joblib
import argparse
import random

SAMPLE_RATE = 16000
DURATION = 5
MAX_TIME_STEPS = 157
N_MELS = 128
N_MFCC = 20

MODEL_PATH = "models/audio_model.keras"
SCALER_PATH = "models/scaler.joblib"
AUGMENTATION = True
HEAVY_AUGMENTATION = True
SILENCE_THRESHOLD = 0.002  # Reject files with energy below this

#### Manual Focal Loss Compatible with TensorFlow 2.17.0 ####
class CustomFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        loss = alpha_t * tf.pow(1 - p_t, self.gamma) * bce

        return tf.reduce_mean(loss)

def pad_or_truncate(feature, max_len):
    if feature.shape[1] < max_len:
        return np.pad(feature, ((0, 0), (0, max_len - feature.shape[1])), mode='constant')
    else:
        return feature[:, :max_len]

def normalize_volume(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def augment_audio(y, sr):
    if random.random() < 0.4:
        noise = np.random.normal(0, 0.002, size=y.shape)
        y = y + noise
    if random.random() < 0.3:
        reverb = np.convolve(y, np.random.rand(1000) * 0.005, mode='full')[:len(y)]
        y = reverb / np.max(np.abs(reverb))

    if HEAVY_AUGMENTATION:
        if random.random() < 0.3:
            comp_ratio = random.uniform(1.5, 3.0)
            threshold = 0.1
            y = np.sign(y) * np.minimum(np.abs(y), threshold + (np.abs(y) - threshold) / comp_ratio)
        if random.random() < 0.3:
            rate = random.uniform(0.97, 1.03)
            y = librosa.effects.time_stretch(y, rate=rate)
        if random.random() < 0.3:
            n_steps = random.uniform(-0.3, 0.3)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y

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

def plot_features(features, title_prefix=""):
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    titles = ["Mel Spectrogram (dB)", "MFCC", "Delta MFCC", "Pitch (F0)", "Energy"]

    for i in range(5):
        im = axes[i].imshow(features[i], aspect='auto', origin='lower', interpolation='none')
        axes[i].set_title(f"{title_prefix} {titles[i]}")
        fig.colorbar(im, ax=axes[i], orientation='vertical')

    plt.tight_layout()
    plt.show()

def predict_single_file(file_path, model, scaler, threshold):
    try:
        y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        y_audio = normalize_volume(y_audio)
        if AUGMENTATION:
            y_audio = augment_audio(y_audio, sr)

        features, avg_energy = extract_combined_features(y_audio)

        if avg_energy < SILENCE_THRESHOLD:
            raise ValueError("Rejected: Audio appears to be silent or too low energy to analyze.")

        if len(features.shape) != 2:
            raise ValueError("Extracted features have invalid shape.")

        # Plot raw features
        feature_splits = []
        split_index = 0
        lengths = [N_MELS, N_MFCC, N_MFCC, 1, 1]  # Mel, MFCC, Delta, F0, Energy

        for l in lengths:
            feature_splits.append(features[split_index:split_index + l])
            split_index += l

        plot_features(feature_splits, title_prefix=os.path.basename(file_path))

        features = scaler.transform(features.reshape(-1, features.shape[-1]))
        input_data = features[..., np.newaxis][np.newaxis, ...]
        pred_prob = model.predict(input_data).flatten()[0]
        label = "Fake" if pred_prob >= threshold else "Real"
        print(f"[*] Prediction for '{file_path}': {label} (prob={pred_prob:.4f})")
    except Exception as e:
        print(f"[!] Error processing file '{file_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the audio file (.wav or .mp3 or .flac)")
    args = parser.parse_args()

    print("[*] Loading scaler and model...")
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        'CustomFocalLoss': CustomFocalLoss
    })

    # training 8-1: 0.3723
    # training 8-2: 0.3368
    default_threshold = 0.3441 # 0.3723  # Tuned by Youden's J Index
    predict_single_file(args.file, model, scaler, default_threshold)
    print("[+] Prediction complete.")
