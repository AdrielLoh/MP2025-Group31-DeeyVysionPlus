# --- Script 1: Preprocess and cache the training data ---
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import multiprocessing
import random
from datetime import datetime
import time
# ===== DO NOT MODIFY UNLESS YOU WANT TO RETRAIN MODEL =====
SAMPLE_RATE = 16000
DURATION = 5
MAX_TIME_STEPS = 157
N_MELS = 128
N_MFCC = 20
# ===== MODIFY AS YOU WISH =====
TRAIN_PATH = "E:/deepfake_training_datasets/VoiceWukong/for-training"
TEST_PATH = "E:/deepfake_training_datasets/training_audio/testing"
REPLACEMENT_REAL_PATH = "E:/deepfake_training_datasets/Common-Voice-Bonafide-Only/clips"
REPLACEMENT_FAKE_PATH = "E:/deepfake_training_datasets/ASVspoof-2021/ASVspoof2021_DF_eval/separated/spoofed"
MODEL_PATH = "models/audio_model.keras"
SCALER_OUTPUT_PATH = "models/scaler.joblib"
TRAIN_CACHE_DIR = "F:/MP-Training-Datasets/audio-bonafide-auged-more/batches/train"
VAL_CACHE_DIR = "F:/MP-Training-Datasets/audio-bonafide-auged-more/batches/val"
PAUSE_FILE = "pause.flag"
BATCH_SIZE = 32
VALID_SPLIT = 0.20

# ===== SCRIPT BEHAVIOURS =====
AUGMENT_TRAINING = True
NEW_SCALER = False
SKIP_INCOMPLETE_BATCH = False
HEAVY_AUGMENTATION = True
SILENCE_THRESHOLD = 0.002

os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
os.makedirs(VAL_CACHE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

def check_pause():
    if os.path.exists(PAUSE_FILE):
        print("[!] Script Paused. Remove 'pause.flag' and the script will resume after 1 minute.")
    while os.path.exists(PAUSE_FILE):
        time.sleep(60)

def pad_or_truncate(feature, max_len):
    if feature.shape[1] < max_len:
        return np.pad(feature, ((0, 0), (0, max_len - feature.shape[1])), mode='constant')
    else:
        return feature[:, :max_len]

def normalize_volume(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def augment_audio(y, sr):
    """
    Augment the audio data with various transformations based on random probabilities and conditions.
    @param y - the audio data
    @param sr - the sample rate of the audio data
    @return The augmented audio data
    """
    if random.random() < 0.4:
        noise = np.random.normal(0, 0.002, size=y.shape) # Noise - Tune values to adjust strength
        y += noise
    if random.random() < 0.4:
        impulse = np.random.rand(min(1000, len(y))) * 0.005
        reverb = np.convolve(y, impulse, mode='full')[:len(y)] # Reverb
        y = reverb / np.max(np.abs(reverb))
    if HEAVY_AUGMENTATION:
        if random.random() < 0.25:
            comp_ratio = random.uniform(1.5, 3.0) # Tune values to adjust strength
            threshold = 0.1
            y = np.sign(y) * np.minimum(np.abs(y), threshold + (np.abs(y) - threshold) / comp_ratio) # Compression
        if random.random() < 0.25:
            rate = random.uniform(0.97, 1.03) # Tune values to adjust strength
            y = librosa.effects.time_stretch(y, rate=rate) # Time stretching
        if random.random() < 0.25:
            n_steps = random.uniform(-0.3, 0.3) # Tune values to adjust strength
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps) # Pitch shifting
    y = np.clip(y, -1.0, 1.0)
    return y

def extract_combined_features(y, sr=SAMPLE_RATE):
    """
    Extract various audio features from the input audio signal.
    @param y - the input audio signal
    @param sr - the sample rate of the audio signal (default is SAMPLE_RATE)
    @returns a tuple containing the extracted features and the mean energy of the audio signal
    """
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

def extract_windows(y_audio, window_size, step_size):
    """
    Extracts 5 second windows from longer audio
    """
    windows = []
    for start in range(0, len(y_audio) - window_size + 1, step_size):
        windows.append(y_audio[start:start+window_size])
    return windows

def get_file_label_pairs(path):
    """
    Generate pairs of file paths and their corresponding labels from the given path.
    @param path - The directory path containing the files
    @return Two lists of file-label pairs for two classes
    """
    class0, class1 = [], []
    for class_dir in ["real", "fake"]:
        label = 0 if class_dir == "real" else 1
        class_path = os.path.join(path, class_dir)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                (class0 if label == 0 else class1).append((os.path.join(class_path, filename), label))
    random.shuffle(class0)
    random.shuffle(class1)
    return class0, class1

def read_processed_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return set(line.strip() for line in f)
    return set()

def append_to_processed_log(log_path, files):
    with open(log_path, "a") as f:
        for file in files:
            f.write(file + "\n")

def get_next_batch_index(cache_dir):
    """
    Get the index for the next batch of data to be processed based on the existing cached files in the directory.
    @param cache_dir - The directory where the cached files are stored.
    @return The index for the next batch to be processed.
    """
    import re
    indices = []
    for fname in os.listdir(cache_dir):
        match = re.match(r'X_batch_(\d+)\.npy', fname)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices) + 1
    else:
        return 0

def get_replacement_files(path, label):
    replacements = []
    for filename in os.listdir(path):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            replacements.append((os.path.join(path, filename), label))
    random.shuffle(replacements)
    return replacements

get_replacement_real_files = lambda: get_replacement_files(REPLACEMENT_REAL_PATH, 0)
get_replacement_fake_files = lambda: get_replacement_files(REPLACEMENT_FAKE_PATH, 1)

def process_single_file(args):
    """
    Process a single audio file by extracting features and labels for training.
    @param args - a tuple containing file_path, label, and scaler
    @return feature_label_pairs - a list of tuples containing features and labels
    """
    file_path, label, scaler = args
    try:
        y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        window_length = int(DURATION * SAMPLE_RATE)
        feature_label_pairs = []
        # If longer than window, extract multiple windows
        if len(y_audio) > window_length:
            step_size = window_length # Set step_size here for overlap (window_length//2 for 50% overlap, window_length for none)
            windows = extract_windows(y_audio, window_length, step_size)
        else:
            windows = [y_audio]
        for window in windows:
            window = normalize_volume(window)
            # Augmentation
            if label == 0 and random.random() < 0.35:
                window = augment_audio(window, sr)
            elif label == 1 and random.random() < 0.25:
                window = augment_audio(window, sr)
            features, avg_energy = extract_combined_features(window)
            if avg_energy < SILENCE_THRESHOLD: # Skips the window if the audio is below the silence threshold
                continue
            features = scaler.transform(features)
            feature_label_pairs.append((features[..., np.newaxis], label))
        if not feature_label_pairs:
            raise Exception("No valid windows from sample")
        return feature_label_pairs
    except Exception as e:
        print(f"[*] Failed to process {file_path}: {e}")
        return None

def preprocess_and_cache_balanced(real_samples, fake_samples, scaler, output_dir, processed_real, processed_fake, log_real, log_fake):
    """
    Preprocess and cache real and fake samples for training.
    @param real_samples - List of real samples
    @param fake_samples - List of fake samples
    @param scaler - Scaler for normalizing samples
    @param output_dir - Directory to save processed samples
    @param processed_real - List of processed real samples
    @param processed_fake - List of processed fake samples
    @param log_real - Log file for real samples
    @param log_fake - Log file for fake samples
    """
    # ===== 1) Get batch index and replacement files =====
    batch_index = get_next_batch_index(output_dir)
    replacement_real_pool = get_replacement_real_files()
    replacement_fake_pool = get_replacement_fake_files()
    used_replacements_real = 0
    used_replacements_fake = 0
    
    """ 2) Filter out processed real and fake samples, then determine the minimum class size between the real and fake samples.
    Iterate over the samples in batch sizes, ensuring replacements are used when necessary. If `SKIP_INCOMPLETE_BATCH` is enabled, skip incomplete batches.
    """
    real_samples = [pair for pair in real_samples if pair[0] not in processed_real]
    fake_samples = [pair for pair in fake_samples if pair[0] not in processed_fake]
    min_class_size = min(len(real_samples), len(fake_samples))
    for i in range(0, min_class_size, BATCH_SIZE // 2):
        check_pause()
        real_batch = real_samples[i:i + BATCH_SIZE // 2]
        fake_batch = fake_samples[i:i + BATCH_SIZE // 2]
        while len(real_batch) < BATCH_SIZE // 2 and used_replacements_real < len(replacement_real_pool):
            real_batch.append(replacement_real_pool[used_replacements_real])
            used_replacements_real += 1
        while len(fake_batch) < BATCH_SIZE // 2 and used_replacements_fake < len(replacement_fake_pool):
            fake_batch.append(replacement_fake_pool[used_replacements_fake])
            used_replacements_fake += 1
        if SKIP_INCOMPLETE_BATCH and (len(real_batch) < BATCH_SIZE // 2 or len(fake_batch) < BATCH_SIZE // 2):
            continue
        
        # ===== Process each file in the batch =====
        def try_processing(batch):
            with multiprocessing.Pool(processes=4) as pool:
                results = pool.map(process_single_file, [(fp, lbl, scaler) for fp, lbl in batch])
            flat_results = []
            for r in results:
                if r:
                    flat_results.extend(r)
            return flat_results
            
        """
        Process real and fake batches of data, ensuring that each batch has a sufficient number of samples. If a batch is incomplete, 
        retry with replacement samples until the batch is filled. 
        Shuffle the data within each batch and save the X and y arrays as numpy files. Update logs with the processed file paths.
        """
        real_results = try_processing(real_batch)
        fake_results = try_processing(fake_batch)
        X_real, y_real = zip(*[res for res in real_results if res is not None]) if any(real_results) else ([], [])
        X_real, y_real = list(X_real), list(y_real)
        X_fake, y_fake = zip(*[res for res in fake_results if res is not None]) if any(fake_results) else ([], [])
        X_fake, y_fake = list(X_fake), list(y_fake)
        if len(X_real) < BATCH_SIZE // 2:
            retry_needed = BATCH_SIZE // 2 - len(X_real)
            retry_batch = replacement_real_pool[used_replacements_real:used_replacements_real + retry_needed]
            used_replacements_real += len(retry_batch)
            retry_results = try_processing(retry_batch)
            for res in retry_results:
                if res:
                    X_real.append(res[0])
                    y_real.append(res[1])
        if len(X_fake) < BATCH_SIZE // 2:
            retry_needed = BATCH_SIZE // 2 - len(X_fake)
            retry_batch = replacement_fake_pool[used_replacements_fake:used_replacements_fake + retry_needed]
            used_replacements_fake += len(retry_batch)
            retry_results = try_processing(retry_batch)
            for res in retry_results:
                if res:
                    X_fake.append(res[0])
                    y_fake.append(res[1])
        if len(X_real) < BATCH_SIZE // 2 or len(X_fake) < BATCH_SIZE // 2:
            continue
        X_batch = list(X_real) + list(X_fake)
        y_batch = list(y_real) + list(y_fake)
        combined = list(zip(X_batch, y_batch))
        random.shuffle(combined)
        X_batch, y_batch = zip(*combined)
        # ===== Save batch =====
        np.save(os.path.join(output_dir, f"X_batch_{batch_index}.npy"), np.array(X_batch))
        np.save(os.path.join(output_dir, f"y_batch_{batch_index}.npy"), np.array(y_batch))
        # ===== Log processed files =====
        real_files_in_batch = [fp for fp, _ in real_batch]
        fake_files_in_batch = [fp for fp, _ in fake_batch]
        append_to_processed_log(log_real, real_files_in_batch)
        append_to_processed_log(log_fake, fake_files_in_batch)
        batch_index += 1

def extract_for_scaler(pair):
    # Performs preprocessing on a subset of data for the scaler fitting
    try:
        y_audio, _ = librosa.load(pair[0], sr=SAMPLE_RATE, mono=True, duration=DURATION)
        features, avg_energy = extract_combined_features(y_audio)
        if avg_energy < SILENCE_THRESHOLD:
            return None
        return features
    except:
        return None

if __name__ == "__main__":
    # ===== MAIN PIPELINE =====
    print("[*] Current Options")
    print(f"[1] AUGMENT_TRAINING = {AUGMENT_TRAINING}")
    print(f"[2] NEW_SCALER = {NEW_SCALER}")
    print(f"[3] SKIP_INCOMPLETE_BATCH = {SKIP_INCOMPLETE_BATCH}")
    print(f"[4] HEAVY_AUGMENTATION = {HEAVY_AUGMENTATION}")
    cont = input("\nConfirm options and continue? (y/n)")
    if cont.lower() != "y":
        print("Exiting script now")
        exit()

    check_pause()
    # ===== Getting all the files from real / fake folders =====
    real_pairs, fake_pairs = get_file_label_pairs(TRAIN_PATH)
    print("\n[*] Running preprocessing_script.py")
    print("[*] Splitting data into training and validation sets...")
    # ===== Splitting the files into two sets =====
    val_real_size = int(len(real_pairs) * VALID_SPLIT)
    val_fake_size = int(len(fake_pairs) * VALID_SPLIT)
    val_real = real_pairs[:val_real_size]
    val_fake = fake_pairs[:val_fake_size]
    train_real_pairs = real_pairs[val_real_size:]
    train_fake_pairs = fake_pairs[val_fake_size:]

    # ===== Scaler fitting / loading existing scaler =====
    if NEW_SCALER:
        print("[*] Fitting scaler using a balanced subset of training data...")
        subset_size = min(len(train_real_pairs), len(train_fake_pairs), 17800)
        subset_real = train_real_pairs[:subset_size]
        subset_fake = train_fake_pairs[:subset_size]
        subset = subset_real + subset_fake
        random.shuffle(subset)

        with multiprocessing.Pool(processes=8) as pool:
            results = pool.map(extract_for_scaler, subset)
        subset_features = [r for r in results if r is not None]

        if not subset_features:
            raise ValueError("Scaler fitting failed: No valid feature samples extracted. Sample is either too silent or corrupted")
        
        subset_stack = np.vstack(subset_features)
        scaler = StandardScaler().fit(subset_stack)
        joblib.dump(scaler, SCALER_OUTPUT_PATH)
    else:
        print("[*] Loading saved scaler...")
        scaler = joblib.load(SCALER_OUTPUT_PATH)

    # ===== Starting main preprocessing =====
    PROCESSED_REAL_LOG = os.path.join(TRAIN_CACHE_DIR, "processed_files_real.txt")
    PROCESSED_FAKE_LOG = os.path.join(TRAIN_CACHE_DIR, "processed_files_fake.txt")
    processed_real = read_processed_log(PROCESSED_REAL_LOG)
    processed_fake = read_processed_log(PROCESSED_FAKE_LOG)
    print("[*] Starting balanced training batch preprocessing...")
    preprocess_and_cache_balanced(train_real_pairs, train_fake_pairs, scaler, TRAIN_CACHE_DIR, processed_real, processed_fake, PROCESSED_REAL_LOG, PROCESSED_FAKE_LOG)
    print("[*] Starting validation batch preprocessing...")
    preprocess_and_cache_balanced(val_real, val_fake, scaler, TRAIN_CACHE_DIR, processed_real, processed_fake, PROCESSED_REAL_LOG, PROCESSED_FAKE_LOG)
    print("--- Pre-processing of training and validation data is completed ---")
    print(f"[+] Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")