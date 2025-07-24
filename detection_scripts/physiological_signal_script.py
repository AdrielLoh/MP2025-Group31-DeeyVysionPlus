import os
import cv2
import numpy as np
import subprocess
import mediapipe as mp
import gc
from tensorflow.keras.utils import register_keras_serializable
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import json
import collections

# ========== Configuration ==========
FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'
MODEL_PATH = 'models/physio_deep_evaluated_9ft3.keras'
fake_threshold = 0.49 # By youden's j index

# ---- rPPG/ROI config (must match training) ----
ROI_INDICES = {
    'left_cheek': [207, 216, 206, 203, 129, 209, 126, 47, 121, 120, 119, 118, 117, 111, 116, 123, 147, 187],
    'right_cheek': [350, 277, 355, 429, 279, 331, 423, 426, 436, 427, 411, 376, 352, 345, 340, 346, 347, 348, 349],
    'forehead':    [10, 338, 297, 332, 284, 251, 301, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70, 71, 21, 54, 103, 67, 109],
    'chin':        [43, 202, 210, 169, 150, 149, 176, 148, 152, 377, 400, 378, 379, 394, 430, 422, 273, 335, 406, 313, 18, 83, 182, 106],
    'nose':        [168, 122, 174, 198, 49, 48, 115, 220, 44, 1, 274, 440, 344, 279, 429, 399, 351]
}
WINDOW_SIZE = 150
HOP_SIZE = 75

# ========== Deep Model Utilities ==========
@register_keras_serializable()
def masked_gap(args):
    f, m = args
    seq_len = tf.shape(f)[1]
    m = m[:, :seq_len]
    m = tf.cast(tf.expand_dims(m, -1), f.dtype)
    num = tf.reduce_sum(f * m, axis=1)
    denom = tf.reduce_sum(m, axis=1)
    pooled = num / tf.clip_by_value(denom, 1e-3, tf.float32.max)
    return pooled

def predict_single_window(model, roi_features, window_mask):
    """Predict on a single window of ROI features."""
    if roi_features.ndim == 2:
        roi_features = np.expand_dims(roi_features, 0)
    if window_mask.ndim == 1:
        window_mask = np.expand_dims(window_mask, 0)
    prediction = model([roi_features, window_mask], training=False)
    return float(prediction.numpy().squeeze())

# ========== Face Detection & Tracking ==========
def load_face_net(proto, model):
    return cv2.dnn.readNetFromCaffe(proto, model)

def detect_faces(frame, net, conf=0.6):
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

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def robust_track_faces(all_boxes, max_lost=5, iou_threshold=0.3, max_distance=100):
    """
    Improved IoU and centroid-based tracker to maintain IDs.
    Args:
        all_boxes: list of boxes per frame
        max_lost: maximum frames a track can be lost before removal
        iou_threshold: minimum IoU for considering a match
        max_distance: maximum centroid distance for considering a match
    Returns: {face_id: [(frame_idx, box), ...]}
    """
    tracks = {}
    active_tracks = {}  # face_id: [last_frame, last_box, lost_count]
    face_id_counter = 0
    for frame_idx, boxes in enumerate(all_boxes):
        # Handle empty frame
        if not boxes:
            # Increment lost count for all active tracks
            for tid in list(active_tracks.keys()):
                active_tracks[tid][2] += 1
                if active_tracks[tid][2] > max_lost:
                    del active_tracks[tid]
            continue
        # Handle first frame or no active tracks
        if not active_tracks:
            for b in boxes:
                tracks[face_id_counter] = [(frame_idx, b)]
                active_tracks[face_id_counter] = [frame_idx, b, 0]
                face_id_counter += 1
            continue
        # Get current track info
        track_ids = list(active_tracks.keys())
        track_boxes = np.array([active_tracks[tid][1] for tid in track_ids])
        # Compute cost matrix using both IoU and centroid distance
        n_tracks = len(track_ids)
        n_boxes = len(boxes)
        cost_matrix = np.ones((n_tracks, n_boxes)) * 1000  # High cost for no match
        # Calculate centroids
        box_centroids = np.array([[x + w/2, y + h/2] for (x, y, w, h) in boxes])
        track_centroids = np.array([[b[0] + b[2]/2, b[1] + b[3]/2] for b in track_boxes])
        # Compute distances
        distances = cdist(track_centroids, box_centroids)
        # Compute IoUs and combined costs
        for i, track_box in enumerate(track_boxes):
            for j, box in enumerate(boxes):
                iou_score = iou(track_box, box)
                distance = distances[i, j]
                # Only consider assignment if IoU is above threshold OR distance is very small
                if iou_score > iou_threshold or distance < max_distance:
                    # Combined cost: weighted sum of (1-IoU) and normalized distance
                    # Lower cost is better
                    iou_cost = 1 - iou_score
                    dist_cost = distance / max_distance
                    cost_matrix[i, j] = 0.6 * iou_cost + 0.4 * dist_cost
        # Solve assignment problem using Hungarian algorithm
        if n_tracks > 0 and n_boxes > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_boxes = set()
            # Process assignments
            for row, col in zip(row_indices, col_indices):
                # Only accept assignment if cost is reasonable
                if cost_matrix[row, col] < 0.9:  # Threshold for valid match
                    tid = track_ids[row]
                    box = boxes[col]
                    # Update track
                    tracks.setdefault(tid, []).append((frame_idx, box))
                    active_tracks[tid] = [frame_idx, box, 0]  # Reset lost count
                    matched_boxes.add(col)
                else:
                    # Poor match - increment lost count
                    tid = track_ids[row]
                    active_tracks[tid][2] += 1
            # Handle unmatched tracks
            for i, tid in enumerate(track_ids):
                if i not in row_indices or cost_matrix[i, col_indices[list(row_indices).index(i)]] >= 0.9:
                    if active_tracks[tid][0] != frame_idx:  # Not updated this frame
                        active_tracks[tid][2] += 1
            # Create new tracks for unmatched boxes
            for j, box in enumerate(boxes):
                if j not in matched_boxes:
                    tracks[face_id_counter] = [(frame_idx, box)]
                    active_tracks[face_id_counter] = [frame_idx, box, 0]
                    face_id_counter += 1
        # Clean up lost tracks
        for tid in list(active_tracks.keys()):
            if active_tracks[tid][2] > max_lost:
                del active_tracks[tid]
    return tracks

def letterbox_resize(frame, target_size=(640, 360), pad_color=0):
    """
    Resize image to fit within target_size, padding to maintain aspect ratio.
    Returns: padded_image, scaling_factor, padding_offsets
    """
    h, w = frame.shape[:2]
    tw, th = target_size

    # Compute scaling factor (fit to the smallest side)
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Compute padding
    pad_w = tw - new_w
    pad_h = th - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad to target size
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded, scale, (left, top)

# ========== rPPG and Signal Extraction ==========
_thread_local = {}
def get_facemesh():
    if "mp_face_mesh" not in _thread_local:
        _thread_local["mp_face_mesh"] = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5
        )
    return _thread_local["mp_face_mesh"]

def extract_landmarks(frame, box):
    """
    Extract 468 facial landmarks (pixel coordinates) for the face inside 'box' in 'frame' using MediaPipe FaceMesh.
    Returns: list of (x, y) tuples (length 468), or None if detection fails.
    """
    try:
        face_mesh = get_facemesh()
        x, y, w, h = box
        roi = frame[y:y+h, x:x+w]
        if roi.shape[0] < 40 or roi.shape[1] < 40:
            return None
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_roi)
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            pts = []
            for lm in lms:
                lx = int(lm.x * w) + x
                ly = int(lm.y * h) + y
                pts.append((lx, ly))
            if len(pts) == 468:
                return pts
        return None
    except Exception:
        return None

def extract_roi_means(frame, box, landmarks, rois=ROI_INDICES):
    """
    Extract mean RGB values for each ROI using MediaPipe landmarks.
    Fixed to handle None landmarks and proper coordinate mapping.
    """
    rgb_means = {}
    if landmarks is None:
        return {roi: [0.0, 0.0, 0.0] for roi in rois}
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]
    for roi, idxs in rois.items():
        pts = []
        for i in idxs:
            if i < len(landmarks):
                lx, ly = landmarks[i]
                lx = max(0, min(lx, frame_w - 1))
                ly = max(0, min(ly, frame_h - 1))
                pts.append([lx, ly])
        pts = np.array(pts, np.int32)
        if len(pts) > 2:
            mask = np.zeros((frame_h, frame_w), np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            vals = []
            for ch in range(3):
                channel_vals = frame[:, :, ch][mask == 1]
                vals.append(float(np.mean(channel_vals)) if len(channel_vals) > 0 else 0.0)
            rgb_means[roi] = vals
        else:
            rgb_means[roi] = [0.0, 0.0, 0.0]
    return rgb_means

from scipy.signal import detrend, butter, filtfilt

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

from sklearn.decomposition import PCA
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

def apply_signal_preprocessing(signal, fs=30, apply_filtering=True):
    """
    Apply light preprocessing to rPPG signals to improved normalization
    """
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

def segment_windows(nframes, win=WINDOW_SIZE, hop=HOP_SIZE):
    return [(s, min(s + win, nframes)) for s in range(0, max(nframes - win + 1, 1), hop)]

# ========== Bounding Box Output & Graphing Drawing ==========
def draw_output(frame, face_box, prediction, confidence, track_id):
    x, y, w, h = face_box
    color = (0, 255, 0) if prediction == 'REAL' else (0, 0, 255)
    label = f"Face {track_id}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y_text = y + h + 25
    cv2.putText(frame, f"{prediction} ({confidence:.2f})", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def draw_rois_on_frame(frame, landmarks, alpha=0.36):
    """
    Overlay translucent polygons for each ROI on the frame using the provided landmarks.
    alpha: 0 (fully transparent) to 1 (fully opaque)
    """
    overlay = frame.copy()
    color_map = {
        'left_cheek': (245, 218, 218),
        'right_cheek': (245, 218, 218),
        'forehead': (245, 218, 218),
        'chin': (245, 218, 218),
        'nose': (245, 218, 218)
    }
    if landmarks is None:
        return frame
    for roi, idxs in ROI_INDICES.items():
        pts = []
        for i in idxs:
            if i < len(landmarks):
                pts.append(landmarks[i])
        pts = np.array(pts, np.int32)
        if len(pts) > 2:
            cv2.polylines(overlay, [pts], isClosed=True, color=color_map.get(roi, (255,255,255)), thickness=2)
            cv2.fillPoly(overlay, [pts], color_map.get(roi, (255,255,255)))
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def plot_rppg_rois(roi_signals_dict, track_id, video_tag, save_dir="static/results"):
    """
    Plots 5 graphs (one for each ROI), each showing the CHROM, POS, ICA signals.
    Returns a dict: {roi_name: filepath}
    """
    os.makedirs(save_dir, exist_ok=True)
    roi_plot_paths = {}
    colors = {'CHROM': 'blue', 'POS': 'green', 'ICA': 'red'}
    for roi, sigs in roi_signals_dict.items():
        plt.figure(figsize=(8, 4))
        for method, signal in sigs.items():
            plt.plot(signal, label=method, color=colors.get(method, None))
        plt.title(f'Face {track_id} - {roi} rPPG Signals')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"rppg_{roi}_face{track_id}_{video_tag}.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        roi_plot_paths[roi] = plot_path
    return roi_plot_paths

# ========== Main Inference Function ==========
def run_detection(video_path, video_tag, output_path='static/results/physio_deep_output.mp4', method="single"):
    output_path = f'static/results/physio_deep_output_{video_tag}.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_net = load_face_net(FACE_PROTO, FACE_MODEL)
    model = load_model(MODEL_PATH)
    window_size = WINDOW_SIZE

    min_face_area_ratio=0.001 # Larger value = more restrictive face tracking 

    # First pass: collect all boxes per frame (for tracking)
    all_boxes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        padded_frame, scale, (pad_left, pad_top) = letterbox_resize(frame, target_size=(640, 360), pad_color=0)
        boxes = detect_faces(padded_frame, net=face_net)
        # Map boxes back to original frame coordinates
        mapped_boxes = []
        frame_area = frame.shape[0] * frame.shape[1]
        for x, y, w, h in boxes:
            # Remove padding, scale back to original
            x_orig = int((x - pad_left) / scale)
            y_orig = int((y - pad_top) / scale)
            w_orig = int(w / scale)
            h_orig = int(h / scale)
            area = w_orig * h_orig
            if area >= frame_area * min_face_area_ratio:
                mapped_boxes.append((x_orig, y_orig, w_orig, h_orig))
        all_boxes.append(mapped_boxes)
    cap.release()

    # Build tracks (face IDs)
    tracks = robust_track_faces(all_boxes)
    if len(all_boxes) < 75:
        return [{
            'track_id': 0,
            'result': "Invalid",
            'real_count': 0,
            'fake_count': 0,
            'confidence': 0,
        }], None

    # Prepare per-face rolling buffers for live windowed prediction
    frame_count = len(all_boxes)
    roi_buffers = collections.defaultdict(lambda: {roi: collections.deque(maxlen=window_size) for roi in ROI_INDICES})
    prediction_memory = collections.defaultdict(list)  # Store (pred, prob) for rPPG summary and plotting
    roi_rgb_histories = collections.defaultdict(lambda: {roi: [] for roi in ROI_INDICES})

    cap = cv2.VideoCapture(video_path)
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        for track_id, detections in tracks.items():
            # Find detection in this frame
            this_box = None
            for fidx, box in detections:
                if fidx == frame_idx:
                    this_box = box
                    break
            if this_box is not None:
                landmarks = extract_landmarks(frame, this_box)
                roi_rgb = extract_roi_means(frame, this_box, landmarks)
                for roi in ROI_INDICES:
                    roi_buffers[track_id][roi].append(roi_rgb[roi])
                    roi_rgb_histories[track_id][roi].append(roi_rgb[roi])  # For full-signal plotting after
                # Draw ROI polygons
                if landmarks is not None:
                    frame = draw_rois_on_frame(frame, landmarks, alpha=0.36)

                # --- Live sliding window prediction ---
                if len(roi_buffers[track_id][ROI_INDICES.keys().__iter__().__next__()]) == window_size:
                    # Build features for the window
                    all_roi_features = []
                    for roi in ROI_INDICES:
                        rgb = np.array(roi_buffers[track_id][roi])
                        chrom = apply_signal_preprocessing(rppg_chrom(rgb), fs=fps)
                        pos   = apply_signal_preprocessing(rppg_pos(rgb), fs=fps)
                        ica   = apply_signal_preprocessing(rppg_pca(rgb), fs=fps)
                        chrom, _ = pad_mask(chrom, window_size)
                        pos, _   = pad_mask(pos, window_size)
                        ica, _   = pad_mask(ica, window_size)
                        roi_feat = np.stack([chrom, pos, ica], axis=-1)
                        all_roi_features.append(roi_feat)
                    X_roi_combined = np.concatenate(all_roi_features, axis=-1)
                    X_mask = np.ones(window_size, dtype=np.float32)
                    prob = predict_single_window(model, X_roi_combined, X_mask)
                    pred = 'FAKE' if prob > fake_threshold else 'REAL'
                else:
                    prob = 0
                    pred = 'REAL'  # Default or "Unknown"

                # Save for rPPG summary/plotting
                prediction_memory[track_id].append((pred, prob))

                # Draw prediction/box on this frame
                frame = draw_output(frame, this_box, pred, prob, track_id)
            else:
                # If the face is missing in this frame, add zeros to history only for rPPG plotting
                for roi in ROI_INDICES:
                    roi_buffers[track_id][roi].append([0.0, 0.0, 0.0])
                    roi_rgb_histories[track_id][roi].append([0.0, 0.0, 0.0])
                prediction_memory[track_id].append(('Unknown', 0))

        # Write annotated frame immediately!
        out.write(frame)

    cap.release()
    out.release()

    # --- Prediction stats and rPPG plotting after video ---
    face_results = []
    roi_signal_plot_paths = {}
    for track_id in tracks:
        preds = [p for p, _ in prediction_memory[track_id] if p in ('REAL', 'FAKE')]
        probs = [prob for _, prob in prediction_memory[track_id] if prob != 0]
        n_real = preds.count('REAL')
        n_fake = preds.count('FAKE')
        result = 'Real' if n_real > n_fake else 'Fake'
        avg_conf = float(np.mean(probs)) if probs else 0.0

        # rPPG signals for full video
        roi_signals_dict_full = {}
        roi_rgb_dict = roi_rgb_histories[track_id]
        for roi in ROI_INDICES:
            rgb = np.array(roi_rgb_dict[roi])
            chrom = apply_signal_preprocessing(rppg_chrom(rgb), fs=fps)
            pos   = apply_signal_preprocessing(rppg_pos(rgb), fs=fps)
            ica   = apply_signal_preprocessing(rppg_pca(rgb), fs=fps)
            roi_signals_dict_full[roi] = {'CHROM': chrom, 'POS': pos, 'ICA': ica}
        roi_plot_paths = plot_rppg_rois(roi_signals_dict_full, track_id, video_tag)
        roi_signal_plot_paths[track_id] = roi_plot_paths

        face_results.append({
            'track_id': track_id,
            'result': result,
            'real_count': n_real,
            'fake_count': n_fake,
            'confidence': round(avg_conf * 100),
        })

    # Attach plots
    for face_result in face_results:
        tid = face_result['track_id']
        face_result['roi_signal_plots'] = roi_signal_plot_paths.get(tid, {})

    # re-encode as H.264 faststart using ffmpeg
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

    if method != "multi":
        if os.path.exists(video_path):
            os.remove(video_path)

    if "mp_face_mesh" in _thread_local:
        try:
            _thread_local["mp_face_mesh"].close()
            del _thread_local["mp_face_mesh"]
        except Exception as e:
            print("Warning: MediaPipe FaceMesh cleanup failed:", e)
    gc.collect()

    return face_results, output_path

