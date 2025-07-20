import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from scipy.signal import detrend, butter, filtfilt
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import subprocess
import gc
from tensorflow.keras.utils import register_keras_serializable
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ========= CONFIG ===========
FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'
MODEL_PATH = 'models/physio_deep_v2.keras'
WINDOW_SIZE = 150
HOP_SIZE = 75
FAKE_THRESHOLD = 0.2686 # Youden's J index

# ========= UTILS ===========
@register_keras_serializable()
def masked_mean(inputs):
    features, mask = inputs
    mask = tf.cast(mask, tf.float32)  # [batch, 150]
    mask = tf.expand_dims(mask, axis=-1)  # [batch, 150, 1]
    features = features * mask
    summed = tf.reduce_sum(features, axis=1)
    denom = tf.reduce_sum(mask, axis=1) + 1e-8
    return summed / denom  # [batch, features]
def load_face_net(proto, model):
    return cv2.dnn.readNetFromCaffe(proto, model)

def detect_faces(frame, net, conf=0.75):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence > conf:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def robust_track_faces(all_boxes, max_lost=5, iou_threshold=0.3, max_distance=100):
    tracks = {}
    active_tracks = {}
    face_id_counter = 0
    for frame_idx, boxes in enumerate(all_boxes):
        if not boxes:
            for tid in list(active_tracks.keys()):
                active_tracks[tid][2] += 1
                if active_tracks[tid][2] > max_lost:
                    del active_tracks[tid]
            continue
        if not active_tracks:
            for b in boxes:
                tracks[face_id_counter] = [(frame_idx, b)]
                active_tracks[face_id_counter] = [frame_idx, b, 0]
                face_id_counter += 1
            continue
        track_ids = list(active_tracks.keys())
        track_boxes = np.array([active_tracks[tid][1] for tid in track_ids])
        n_tracks = len(track_ids)
        n_boxes = len(boxes)
        cost_matrix = np.ones((n_tracks, n_boxes)) * 1000
        box_centroids = np.array([[x + w/2, y + h/2] for (x, y, w, h) in boxes])
        track_centroids = np.array([[b[0] + b[2]/2, b[1] + b[3]/2] for b in track_boxes])
        distances = cdist(track_centroids, box_centroids)
        for i, track_box in enumerate(track_boxes):
            for j, box in enumerate(boxes):
                iou_score = iou(track_box, box)
                distance = distances[i, j]
                if iou_score > iou_threshold or distance < max_distance:
                    iou_cost = 1 - iou_score
                    dist_cost = distance / max_distance
                    cost_matrix[i, j] = 0.6 * iou_cost + 0.4 * dist_cost
        if n_tracks > 0 and n_boxes > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_boxes = set()
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.9:
                    tid = track_ids[row]
                    box = boxes[col]
                    tracks.setdefault(tid, []).append((frame_idx, box))
                    active_tracks[tid] = [frame_idx, box, 0]
                    matched_boxes.add(col)
                else:
                    tid = track_ids[row]
                    active_tracks[tid][2] += 1
            for i, tid in enumerate(track_ids):
                if i not in row_indices or cost_matrix[i, col_indices[list(row_indices).index(i)]] >= 0.9:
                    if active_tracks[tid][0] != frame_idx:
                        active_tracks[tid][2] += 1
            for j, box in enumerate(boxes):
                if j not in matched_boxes:
                    tracks[face_id_counter] = [(frame_idx, box)]
                    active_tracks[face_id_counter] = [frame_idx, box, 0]
                    face_id_counter += 1
        for tid in list(active_tracks.keys()):
            if active_tracks[tid][2] > max_lost:
                del active_tracks[tid]
    return tracks

def extract_face_rgb_mean(frame, box):
    x, y, w, h = box
    x1, y1, x2, y2 = max(0, x), max(0, y), min(x+w, frame.shape[1]), min(y+h, frame.shape[0])
    face_region = frame[y1:y2, x1:x2]
    if face_region.size == 0:
        return [0.0, 0.0, 0.0]
    mean_rgb = np.mean(np.mean(face_region, axis=0), axis=0)
    return mean_rgb.tolist()

def rppg_chrom(rgb):
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] < 10:
        return np.zeros(S.shape[0])
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    h = S[:, 1] - S[:, 0]
    s = S[:, 1] + S[:, 0] - 2 * S[:, 2]
    return (h + s).astype(np.float32)

def rppg_pos(rgb):
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] < 10:
        return np.zeros(S.shape[0])
    S_norm = S / (S.mean(0) + 1e-6)
    P = np.array([[0, 1, -1], [-2, 1, 1]], np.float32)
    Y = S_norm @ P.T
    std_y = Y.std(0) + 1e-6
    Y_norm = Y / std_y
    alpha = Y_norm[:, 0].std() / (Y_norm[:, 1].std() + 1e-6)
    rppg = Y_norm[:, 0] - alpha * Y_norm[:, 1]
    return rppg.astype(np.float32)

def rppg_pca(rgb):
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] < 10:
        return rppg_chrom(rgb)
    pca = PCA(n_components=3)
    X = S - S.mean(0)
    try:
        S_ = pca.fit_transform(X)
        idx = np.argmax(S_.var(axis=0))
        return S_[:, idx].astype(np.float32)
    except:
        return rppg_chrom(rgb)

def rppg_green(rgb):
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[0] < 10:
        return np.zeros(S.shape[0])
    return S[:,1].astype(np.float32)

def rppg_pbv(rgb):
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] < 10:
        return np.zeros(S.shape[0])
    S_norm = S / (S.mean(0) + 1e-6)
    P = np.array([[0, 1, -1], [-2, 1, 1]], np.float32)
    Y = S_norm @ P.T
    Y_centered = Y - Y.mean(axis=0)
    pbv = Y_centered[:,0] - Y_centered[:,1]
    return pbv.astype(np.float32)

def apply_signal_preprocessing(signal, fs=30, apply_filtering=True):
    signal = np.array(signal, dtype=np.float32)
    if signal.size == 0 or len(signal) < 10:
        return signal
    processed = detrend(signal, type='linear').astype(np.float32)
    if apply_filtering and len(signal) > 30:
        filter_order = 3
        min_length_for_filtfilt = 3 * filter_order * 2
        if len(signal) >= min_length_for_filtfilt:
            nyquist = fs / 2.0
            low_cutoff = 0.5 / nyquist
            high_cutoff = 5.0 / nyquist
            low_cutoff = max(0.01, min(low_cutoff, 0.45))
            high_cutoff = max(0.55, min(high_cutoff, 0.99))
            if low_cutoff < high_cutoff and high_cutoff < 1.0 and low_cutoff > 0:
                b, a = butter(filter_order, [low_cutoff, high_cutoff], btype='band')
                padlen = min(len(processed) // 3, filter_order * 3)
                processed = filtfilt(b, a, processed, padlen=padlen).astype(np.float32)
    mean_val = np.mean(processed)
    std_val = np.std(processed)
    if std_val > 1e-6:
        outlier_threshold = 5 * std_val
        processed = np.clip(processed, mean_val - outlier_threshold, mean_val + outlier_threshold)
        median_val = np.median(processed)
        mad = np.median(np.abs(processed - median_val))
        if mad > 1e-6:
            processed = ((processed - median_val) / (1.4826 * mad)).astype(np.float32)
        else:
            processed = ((processed - mean_val) / std_val).astype(np.float32)
    else:
        processed = np.zeros_like(processed, dtype=np.float32)
    return processed

def pad_mask(arr, win):
    L = len(arr)
    pad = win - L
    mask = np.concatenate([np.ones(L), np.zeros(max(pad, 0))])
    arr = np.concatenate([arr, np.zeros(max(pad, 0))])
    return arr[:win].astype(np.float32), mask[:win].astype(np.float32)

def segment_windows(nframes, win=150, hop=75):
    return [(s, min(s + win, nframes)) for s in range(0, max(nframes - win + 1, 1), hop)]

def draw_output(frame, face_box, prediction, confidence, track_id):
    x, y, w, h = face_box
    color = (0, 255, 0) if prediction == 'REAL' else (0, 0, 255)
    label = f"Face {track_id}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y_text = y + h + 25
    cv2.putText(frame, f"{prediction} ({confidence:.2f})", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ========= MAIN PIPELINE ==========
def run_detection(video_path, video_tag, output_dir="static/results", method="single"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"physio_deep_output_{video_tag}.mp4")
    
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load frames and face boxes
    frames = []
    all_boxes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_faces(frame, net=face_net)
        # --- Face filtering by size ---
        min_face_area_ratio = 0.005
        filtered_boxes = []
        frame_area = frame.shape[0] * frame.shape[1]
        for box in boxes:
            _, _, w, h = box
            area = w * h
            if area >= frame_area * min_face_area_ratio:
                filtered_boxes.append(box)
        all_boxes.append(filtered_boxes)
        frames.append(frame.copy())
    cap.release()
    num_frames = len(frames)
    
    # --- Track faces ---
    tracks = robust_track_faces(all_boxes)
    if len(all_boxes) < 75:  # Track length validation
        return [{
            'track_id': 0,
            'result': "Invalid",
            'real_count': 0,
            'fake_count': 0,
            'confidence': 0,
        }], None

    frame_predictions = [[] for _ in range(num_frames)]

    for track_id, detections in tracks.items():
        per_frame_boxes = [None] * num_frames
        for fidx, box in detections:
            per_frame_boxes[fidx] = box

        windows = segment_windows(num_frames, win=WINDOW_SIZE, hop=HOP_SIZE)
        for start, end in windows:
            frames_win = frames[start:end]
            boxes_win = per_frame_boxes[start:end]
            window_mask = [1 if b is not None else 0 for b in boxes_win]
            rgb_signals = [extract_face_rgb_mean(f, b) if b is not None else [0.0, 0.0, 0.0]
                           for f, b in zip(frames_win, boxes_win)]

            signals = [
                apply_signal_preprocessing(rppg_chrom(rgb_signals), fs=fps),
                apply_signal_preprocessing(rppg_pos(rgb_signals), fs=fps),
                apply_signal_preprocessing(rppg_pca(rgb_signals), fs=fps),
                apply_signal_preprocessing(rppg_green(rgb_signals), fs=fps),
                apply_signal_preprocessing(rppg_pbv(rgb_signals), fs=fps),
            ]
            signals = [pad_mask(sig, WINDOW_SIZE)[0] for sig in signals]
            mask = pad_mask(window_mask, WINDOW_SIZE)[1]
            X_window = np.stack(signals, axis=0).astype(np.float32)
            mask = mask.astype(np.float32)
            # === per-window, per-channel z-score normalization ===
            for c in range(X_window.shape[0]):
                m = np.mean(X_window[c])
                s = np.std(X_window[c])
                if s < 1e-6: s = 1.0
                X_window[c] = (X_window[c] - m) / s

            prob = float(model.predict([X_window[None, ...], mask[None, ...]], verbose=0)[0][0])
            pred = 'FAKE' if prob > FAKE_THRESHOLD else 'REAL'
            for i in range(start, end):
                frame_predictions[i].append((track_id, pred, prob, per_frame_boxes[i]))

    # Write video with predictions
    for frame_idx, frame in enumerate(frames):
        faces_in_frame = {}
        for track_id, pred, prob, box in frame_predictions[frame_idx]:
            if box is not None:
                if track_id not in faces_in_frame:
                    faces_in_frame[track_id] = {'probs': [], 'boxes': [], 'preds': []}
                faces_in_frame[track_id]['probs'].append(prob)
                faces_in_frame[track_id]['boxes'].append(box)
                faces_in_frame[track_id]['preds'].append(pred)
        for track_id in faces_in_frame:
            probs = np.array(faces_in_frame[track_id]['probs'])
            median_prob = float(np.median(probs))
            label_pred = 'FAKE' if median_prob > FAKE_THRESHOLD else 'REAL'
            box = faces_in_frame[track_id]['boxes'][0]
            frame = draw_output(frame, box, label_pred, median_prob, track_id)
        out.write(frame)
    out.release()

    # FFmpeg re-encode for compatibility
    fixed_output_path = output_path.replace('.mp4', f'_fixed.mp4')
    converted_video = subprocess.run([
        'ffmpeg', '-y', '-i', output_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart',
        fixed_output_path
    ])
    if converted_video.returncode != 0:
        fixed_output_path = output_path

    if os.path.exists(output_path):
        os.remove(output_path)
    output_path = fixed_output_path

    # Plot rPPG signals for each face
    rppg_plots = {}
    for track_id, detections in tracks.items():
        per_frame_boxes = [None] * num_frames
        for fidx, box in detections:
            per_frame_boxes[fidx] = box
        rgb_hist = [extract_face_rgb_mean(f, b) if b is not None else [0.0, 0.0, 0.0]
                    for f, b in zip(frames, per_frame_boxes)]
        signals_dict = {
            'chrom': apply_signal_preprocessing(rppg_chrom(rgb_hist), fs=fps),
            'pos': apply_signal_preprocessing(rppg_pos(rgb_hist), fs=fps),
            'ica': apply_signal_preprocessing(rppg_pca(rgb_hist), fs=fps),
            'green': apply_signal_preprocessing(rppg_green(rgb_hist), fs=fps),
            'pbv': apply_signal_preprocessing(rppg_pbv(rgb_hist), fs=fps),
        }
        plt.figure(figsize=(10,5))
        for name, sig in signals_dict.items():
            plt.plot(sig, label=name)
        plt.title(f'Face {track_id} rPPG Signals')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"face_{track_id}_rppg_plot_{video_tag}.png")
        plt.savefig(plot_path)
        plt.close()
        rppg_plots[track_id] = plot_path

    # --- Generate face_results as in old script ---
    face_results = []
    for track_id, detections in tracks.items():
        # Aggregate frame_predictions to get stats
        preds = []
        probs = []
        for frame_idx in range(num_frames):
            frame_preds = [p for tid, p, prob, _ in frame_predictions[frame_idx] if tid == track_id]
            frame_probs = [prob for tid, p, prob, _ in frame_predictions[frame_idx] if tid == track_id]
            if frame_preds:
                preds.append(frame_preds[0])
                probs.append(frame_probs[0])
        n_real = preds.count('REAL')
        n_fake = preds.count('FAKE')
        result = 'Real' if n_real > n_fake else 'Fake'
        avg_conf = float(np.mean(probs)) if probs else 0.0

        face_results.append({
            'track_id': track_id,
            'result': result,
            'real_count': n_real,
            'fake_count': n_fake,
            'confidence': round(avg_conf * 100),
            'roi_signal_plots': rppg_plots.get(track_id, {}),
        })

    # Resource cleanup (model, gc)
    if method != "multi":
        if os.path.exists(video_path):
            os.remove(video_path)
    try:
        gc.collect()
    except Exception:
        pass

    return face_results, output_path
