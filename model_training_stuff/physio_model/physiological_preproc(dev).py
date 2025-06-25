import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy.signal import detrend, butter, filtfilt, periodogram
import scipy.stats

# === rPPG multi-method and extended feature extraction ===
def rppg_chrom(rgb):
    S = rgb
    h = (S[:, 1] - S[:, 0]) / (np.std(S[:, 1] - S[:, 0]) + 1e-8)
    s = (S[:, 1] + S[:, 0] - 2 * S[:, 2]) / (np.std(S[:, 1] + S[:, 0] - 2 * S[:, 2]) + 1e-8)
    return h + s

def rppg_pos(rgb):
    S = rgb
    S_mean = np.mean(S, axis=0)
    S = S / (S_mean + 1e-8)
    Xcomp = 3 * S[:, 0] - 2 * S[:, 1]
    Ycomp = 1.5 * S[:, 0] + S[:, 1] - 1.5 * S[:, 2]
    return Xcomp / (Ycomp + 1e-8)

def rppg_green(rgb):
    return rgb[:, 1]

def compute_band_powers(f, pxx, bands):
    total_power = np.sum(pxx)
    power_features = []
    for low, high in bands:
        mask = (f >= low) & (f <= high)
        power = np.sum(pxx[mask])
        ratio = power / (total_power + 1e-8)
        power_features.extend([power, ratio])
    return power_features

def compute_autocorrelation(sig):
    if len(sig) < 2:
        return 0
    acf = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode='full')
    acf = acf[acf.size // 2:]
    peak_lag = np.argmax(acf[1:]) + 1 if len(acf) > 1 else 1
    return acf[peak_lag] / (acf[0] + 1e-8) if acf[0] != 0 else 0

def compute_entropy(sig):
    hist, _ = np.histogram(sig, bins=32, density=True)
    hist += 1e-8
    return -np.sum(hist * np.log2(hist))

def compute_rppg_features_multi(rgb_signal, fs):
    rgb_detrend = detrend(rgb_signal, axis=0)
    rgb_norm = (rgb_detrend - np.mean(rgb_detrend, axis=0)) / (np.std(rgb_detrend, axis=0) + 1e-8)
    rppg_methods = {
        "CHROM": rppg_chrom(rgb_norm),
        "POS": rppg_pos(rgb_norm),
        "GREEN": rppg_green(rgb_norm),
    }
    bands = [
        (0.7, 2.5),  # Cardiac
        (0.2, 0.6),  # Respiratory
        (4.0, 8.0),  # Noise
    ]
    all_features = []
    for key, rppg_sig in rppg_methods.items():
        if len(rppg_sig) <= 21:
            all_features.extend([0]*19)
            continue
        sig_filt = butter_bandpass_filter(rppg_sig, fs)
        f, pxx = periodogram(sig_filt, fs)
        valid = (f >= 0.7) & (f <= 4.0)
        f, pxx = f[valid], pxx[valid]
        if len(f) == 0:
            all_features.extend([0]*19)
            continue
        peak_idx = np.argmax(pxx)
        hr_freq, hr_bpm, hr_power = f[peak_idx], f[peak_idx]*60, pxx[peak_idx]
        snr = hr_power / (np.sum(pxx) - hr_power + 1e-8)
        band_powers = compute_band_powers(f, pxx, bands)
        autocorr_peak = compute_autocorrelation(sig_filt)
        entropy = compute_entropy(sig_filt)
        kurt = scipy.stats.kurtosis(sig_filt)
        skew = scipy.stats.skew(sig_filt)
        features = [
            np.mean(sig_filt), np.std(sig_filt), np.max(sig_filt), np.min(sig_filt),
            np.ptp(sig_filt), hr_freq, hr_bpm, hr_power, snr,
            autocorr_peak, entropy, kurt, skew
        ] + band_powers
        all_features.extend(features)
    return np.array(all_features, dtype=np.float32)  # 57 features

# --- Everything below is the same as your original script, except ---
# --- compute_rppg_features is replaced with compute_rppg_features_multi ---

def get_face_net():
    FACE_PROTO = '../../models/weights-prototxt.txt'
    FACE_MODEL = '../../models/res_ssd_300Dim.caffeModel'
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    return face_net

def detect_faces_dnn(frame, face_net, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, boxA[2] * boxA[3])
    boxBArea = max(1, boxB[2] * boxB[3])
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou_val

def track_faces(all_face_boxes, iou_thresh=0.5):
    tracks = []
    next_track_id = 0
    for frame_idx, boxes in enumerate(all_face_boxes):
        assigned = [False] * len(boxes)
        for track in tracks:
            found = False
            for i, box in enumerate(boxes):
                if not assigned[i] and iou(track['last_box'], box) > iou_thresh and frame_idx - track['last_frame'] == 1:
                    track['boxes'].append((frame_idx, box))
                    track['last_box'] = box
                    track['last_frame'] = frame_idx
                    assigned[i] = True
                    found = True
                    break
        for i, box in enumerate(boxes):
            if not assigned[i]:
                tracks.append({'boxes': [(frame_idx, box)], 'last_box': box, 'last_frame': frame_idx, 'id': next_track_id})
                next_track_id += 1
    tracklets = {t['id']: t['boxes'] for t in tracks if len(t['boxes']) > 2}
    return tracklets

def extract_rgb_signal_track(frames, face_boxes):
    r_means, g_means, b_means, face_mask = [], [], [], []
    for frame, box in zip(frames, face_boxes):
        if box is not None:
            x, y, w, h = box
            x, y = max(x, 0), max(y, 0)
            x2 = min(x + w, frame.shape[1])
            y2 = min(y + h, frame.shape[0])
            if x2 > x and y2 > y:
                roi = frame[y:y2, x:x2]
                if roi.size == 0:
                    r_means.append(0)
                    g_means.append(0)
                    b_means.append(0)
                    face_mask.append(0)
                else:
                    b, g, r = cv2.split(roi)
                    r_means.append(np.mean(r))
                    g_means.append(np.mean(g))
                    b_means.append(np.mean(b))
                    face_mask.append(1)
            else:
                r_means.append(0)
                g_means.append(0)
                b_means.append(0)
                face_mask.append(0)
        else:
            r_means.append(0)
            g_means.append(0)
            b_means.append(0)
            face_mask.append(0)
    rgb_signal = np.stack([r_means, g_means, b_means], axis=-1)  # shape: (frames, 3)
    return rgb_signal, np.array(face_mask)

def butter_bandpass_filter(signal, fs, lowcut=0.7, highcut=4.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def sliding_windows_with_mask(signal, mask, window_size, hop_size, min_face_ratio=0.8):
    windows = []
    for start in range(0, len(signal) - window_size + 1, hop_size):
        end = start + window_size
        window_mask = mask[start:end]
        if np.mean(window_mask) >= min_face_ratio:
            windows.append(signal[start:end, :])
    return np.stack(windows) if windows else np.empty((0, window_size, signal.shape[1]))

def preprocess_video_worker(args):
    video_path, window_size, hop_size = args
    face_net = get_face_net()
    cap = cv2.VideoCapture(video_path)
    frames = []
    all_boxes = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))
        boxes = detect_faces_dnn(frame, face_net)
        frames.append(frame)
        all_boxes.append(boxes)
    cap.release()
    if len(frames) < window_size:
        return (None, 0, os.path.basename(video_path), 0, len(frames))
    tracks = track_faces(all_boxes)
    rppg_features_all_tracks = []
    for track_id, track in tracks.items():
        per_frame_boxes = [None] * len(frames)
        for (frame_idx, box) in track:
            per_frame_boxes[frame_idx] = box
        rgb_signal, face_mask = extract_rgb_signal_track(frames, per_frame_boxes)
        rgb_windows = sliding_windows_with_mask(rgb_signal, face_mask, window_size, hop_size, min_face_ratio=0.8)
        for win in rgb_windows:
            rppg_feat = compute_rppg_features_multi(win, fps)
            rppg_features_all_tracks.append(rppg_feat)
    if rppg_features_all_tracks:
        rppg_features_all_tracks = np.stack(rppg_features_all_tracks, axis=0)
        return (
            rppg_features_all_tracks,
            rppg_features_all_tracks.shape[0],
            os.path.basename(video_path),
            None,
            len(frames),
        )
    else:
        return (None, 0, os.path.basename(video_path), 0, len(frames))

def cache_batches_parallel(video_dir, label, class_idx, batch_size=128, cache_dir='cache/batches/train',
                          window_size=150, hop_size=75, max_workers=None):
    os.makedirs(cache_dir, exist_ok=True)
    X_feat, y = [], []
    file_list = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ])
    batch_idx = 0
    tasks = [(video_path, window_size, hop_size) for video_path in file_list]
    if max_workers:
        n_workers = max_workers
    else:
        n_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
    print(f"[INFO] Using {n_workers} workers for {label} ({len(file_list)} videos)")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(
            tqdm(
                executor.map(preprocess_video_worker, tasks),
                total=len(tasks),
                desc=f"Processing {label}"
            )
        )
        for feats, count, vname, _, n_frames in results:
            if feats is None or count == 0:
                print(f"[WARN] {vname} too short for windowing or not enough valid face windows. Skipped.")
                continue
            X_feat.append(feats)
            y.append(np.full((feats.shape[0],), class_idx))
            total_samples = sum([arr.shape[0] for arr in X_feat])
            while total_samples >= batch_size:
                Xf, Y = [], []
                collected = 0
                while X_feat and collected < batch_size:
                    needed = batch_size - collected
                    take = min(needed, X_feat[0].shape[0])
                    Xf.append(X_feat[0][:take])
                    Y.append(y[0][:take])
                    if take < X_feat[0].shape[0]:
                        X_feat[0] = X_feat[0][take:]
                        y[0] = y[0][take:]
                    else:
                        X_feat.pop(0)
                        y.pop(0)
                    collected += take
                Xf = np.concatenate(Xf)
                Y = np.concatenate(Y)
                np.save(os.path.join(cache_dir, f"{label}_Xrppg_batch_{batch_idx}.npy"), Xf)
                np.save(os.path.join(cache_dir, f"{label}_y_batch_{batch_idx}.npy"), Y)
                batch_idx += 1
                total_samples -= batch_size
        if X_feat:
            Xf = np.concatenate(X_feat)
            Y = np.concatenate(y)
            np.save(os.path.join(cache_dir, f"{label}_Xrppg_batch_{batch_idx}.npy"), Xf)
            np.save(os.path.join(cache_dir, f"{label}_y_batch_{batch_idx}.npy"), Y)

if __name__ == "__main__":
    # Example usage: set your dataset paths
    cache_batches_parallel(
        'G:/deepfake_training_datasets/DeeperForensics/validation/real',
        label='real',
        class_idx=0,
        batch_size=64,
        cache_dir='D:/model_training/cache/batches/val/real'
    )
    cache_batches_parallel(
        'G:/deepfake_training_datasets/DeeperForensics/validation/fake',
        label='fake',
        class_idx=1,
        batch_size=64,
        cache_dir='D:/model_training/cache/batches/val/fake'
    )
