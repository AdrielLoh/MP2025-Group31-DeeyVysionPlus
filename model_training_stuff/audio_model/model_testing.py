# --- Script 3: Testing and evaluating the model ---

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import joblib
import multiprocessing

SAMPLE_RATE = 16000
DURATION = 5
MAX_TIME_STEPS = 157
N_MELS = 128
N_MFCC = 20

TEST_PATH = "G:/deepfake_training_datasets/training_audio/testing"
TEST_CACHE_DIR = "D:/model_training/cache/batches/test"
MODEL_PATH = "models/audio_model.keras"
SCALER_PATH = "models/scaler.joblib"
BATCH_SIZE = 32
USE_CACHED = True  # Set False if want to preprocess data from scratch

os.makedirs(TEST_CACHE_DIR, exist_ok=True)

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

def get_file_label_pairs(path):
    file_label_pairs = []
    for class_dir in ["real", "fake"]:
        label = 0 if class_dir == "real" else 1
        class_path = os.path.join(path, class_dir)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.endswith('.wav') or filename.endswith('.mp3') or filename.endswith('.flac'):
                file_label_pairs.append((os.path.join(class_path, filename), label))
    return file_label_pairs

def extract_and_scale(pair):
    file_path, label, scaler = pair
    try:
        y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        y_audio = normalize_volume(y_audio)
        features, _ = extract_combined_features(y_audio)
        features = scaler.transform(features)
        return features[..., np.newaxis], label
    except Exception as e:
        print(f"[*] Failed to process {file_path}: {e}")
        return None

def preprocess_and_cache_test(file_label_pairs, scaler):
    print("[*] Extracting features with multiprocessing...")
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(extract_and_scale, [(f, l, scaler) for f, l in file_label_pairs])

    results = [r for r in results if r is not None]
    X_all = np.array([r[0] for r in results])
    y_all = np.array([r[1] for r in results])

    np.save(os.path.join(TEST_CACHE_DIR, "X_test.npy"), X_all)
    np.save(os.path.join(TEST_CACHE_DIR, "y_test.npy"), y_all)
    return X_all, y_all

if __name__ == "__main__":
    print("[*] Loading scaler and model...")
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
    'CustomFocalLoss': CustomFocalLoss
})

    if USE_CACHED and os.path.exists(os.path.join(TEST_CACHE_DIR, "X_test.npy")):
        print("[*] Loading cached test data...")
        X_test = np.load(os.path.join(TEST_CACHE_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(TEST_CACHE_DIR, "y_test.npy"))
    else:
        print("[*] Preprocessing test data...")
        file_label_pairs = get_file_label_pairs(TEST_PATH)
        X_test, y_test = preprocess_and_cache_test(file_label_pairs, scaler)

    print("[*] Running model prediction...")
    y_pred_probs = model.predict(X_test).flatten()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    youdens_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youdens_j)]
    print(f"[*] Optimal threshold by Youden's J index: {best_threshold:.4f}")

    y_pred = (y_pred_probs >= best_threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure()
    plt.hist(y_pred_probs, bins=50, color='skyblue', edgecolor='black')
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted Probability of Being Fake")
    plt.ylabel("Sample Count")
    plt.grid(True)
    plt.show()

    below_thresh = np.sum(y_pred_probs < best_threshold)
    above_thresh = np.sum(y_pred_probs >= best_threshold)

    plt.figure()
    plt.bar(['(Real)', '(Fake)'], [below_thresh, above_thresh], color=['green', 'red'])
    plt.title("Probability Split: Below vs Above Threshold")
    plt.ylabel("Number of Samples")
    plt.grid(axis='y')
    plt.show()

    real_probs = y_pred_probs[y_test == 0]
    fake_probs = y_pred_probs[y_test == 1]

    plt.figure()
    plt.hist(real_probs, bins=50, alpha=0.6, label='Real', color='green', edgecolor='black')
    plt.hist(fake_probs, bins=50, alpha=0.6, label='Fake', color='red', edgecolor='black')
    plt.title("Prediction Probability Distribution by Class")
    plt.xlabel("Predicted Probability of Being Fake")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("[+] Testing complete.")
