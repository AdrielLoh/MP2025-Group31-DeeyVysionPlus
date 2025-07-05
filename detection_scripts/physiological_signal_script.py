import os
import cv2
import numpy as np
import joblib
from scipy.signal import detrend, butter, filtfilt, periodogram
import scipy.stats
import matplotlib.pyplot as plt
import uuid
import time
import subprocess
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Model and face detection setup
clf = joblib.load('models/physio_detection_xgboost_best.pkl')
scaler = joblib.load('models/physio_scaler.pkl')
FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

def detect_faces_dnn(frame, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
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

def butter_bandpass_filter(signal, fs, lowcut=0.7, highcut=4.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return np.zeros_like(signal)
    return filtfilt(b, a, signal)

def plot_rppg_analysis(rppg_sig, f, pxx, real_count, fake_count, save_dir, track_id, uid, hr_history=None, band_power_features=None, fps=30):
    """
    Enhanced plotting function that generates 6 plots:
    1. rPPG Signal
    2. Frequency Spectrum
    3. Prediction Count
    4. Heart-Rate Trace over time
    5. Heart-Rate Distribution histogram
    6. Band-Power Ratios bar chart
    """
    os.makedirs(save_dir, exist_ok=True)
    plot_id = f"face_{track_id}_{uid}"
    
    # 1. rPPG Signal Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rppg_sig, label='rPPG Signal', color='blue')
    plt.title(f'Face {track_id} - rPPG Time Series', fontsize=14)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    rppg_signal_plot_path = os.path.join(save_dir, f'rppg_signal_{plot_id}.png')
    plt.savefig(rppg_signal_plot_path, dpi=150)
    plt.close()
    
    # 2. Frequency Spectrum Plot
    plt.figure(figsize=(10, 6))
    plt.plot(f, pxx, label='Power Spectrum', color='purple')
    plt.title(f'Face {track_id} - rPPG Frequency Spectrum', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    rppg_spectrum_plot_path = os.path.join(save_dir, f'rppg_spectrum_{plot_id}.png')
    plt.savefig(rppg_spectrum_plot_path, dpi=150)
    plt.close()
    
    # 3. Prediction Count Plot
    plt.figure(figsize=(8, 6))
    plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'], alpha=0.7)
    plt.title(f'Face {track_id} - Counts of Real and Fake Predictions', fontsize=14)
    plt.xlabel('Prediction Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    prediction_count_path = os.path.join(save_dir, f'prediction_count_{plot_id}.png')
    plt.savefig(prediction_count_path, dpi=150)
    plt.close()
    
    # 4. Heart-Rate Trace over time
    hr_trace_plot_path = None
    if hr_history is not None and len(hr_history) > 0:
        # Filter out invalid HR values
        valid_hr = [hr for hr in hr_history if np.isfinite(hr) and hr > 0]
        if len(valid_hr) > 0:
            # Create time axis in seconds
            time_axis = np.arange(len(hr_history)) / fps
            
            plt.figure(figsize=(12, 6))
            plt.plot(time_axis, hr_history, marker='o', markersize=3, linewidth=2, color='red', alpha=0.8)
            plt.title(f'Face {track_id} - Heart Rate Trace Over Time', fontsize=14)
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Heart Rate (BPM)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add mean line
            mean_hr = np.mean(valid_hr)
            plt.axhline(y=mean_hr, color='orange', linestyle='--', alpha=0.7, 
                       label=f'Mean HR: {mean_hr:.1f} BPM')
            plt.legend()
            plt.tight_layout()
            hr_trace_plot_path = os.path.join(save_dir, f'hr_trace_{plot_id}.png')
            plt.savefig(hr_trace_plot_path, dpi=150)
            plt.close()
    
    # 5. Heart-Rate Distribution histogram
    hr_dist_plot_path = None
    if hr_history is not None and len(hr_history) > 0:
        # Filter out invalid HR values
        valid_hr = [hr for hr in hr_history if np.isfinite(hr) and hr > 0]
        if len(valid_hr) > 1:
            plt.figure(figsize=(10, 6))
            plt.hist(valid_hr, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Face {track_id} - Heart Rate Distribution', fontsize=14)
            plt.xlabel('Heart Rate (BPM)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            mean_hr = np.mean(valid_hr)
            std_hr = np.std(valid_hr)
            plt.axvline(x=mean_hr, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean_hr:.1f} ± {std_hr:.1f} BPM')
            plt.legend()
            plt.tight_layout()
            hr_dist_plot_path = os.path.join(save_dir, f'hr_distribution_{plot_id}.png')
            plt.savefig(hr_dist_plot_path, dpi=150)
            plt.close()
    
    # 6. Band-Power Ratios bar chart
    band_power_plot_path = None
    if band_power_features is not None and len(band_power_features) >= 6:
        # Extract band power ratios (every other element starting from index 1)
        band_ratios = band_power_features[1::2]  # [ratio1, ratio2, ratio3]
        band_names = ['0.7-2.5 Hz\n(HR Band)', '0.2-0.6 Hz\n(LF Band)', '4.0-8.0 Hz\n(HF Band)']
        
        if len(band_ratios) >= 3:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(band_names, band_ratios[:3], 
                          color=['lightcoral', 'lightblue', 'lightgreen'], 
                          alpha=0.8, edgecolor='black')
            plt.title(f'Face {track_id} - Band Power Ratios', fontsize=14)
            plt.xlabel('Frequency Bands', fontsize=12)
            plt.ylabel('Power Ratio', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, ratio in zip(bars, band_ratios[:3]):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{ratio:.3f}', ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            band_power_plot_path = os.path.join(save_dir, f'band_power_ratios_{plot_id}.png')
            plt.savefig(band_power_plot_path, dpi=150)
            plt.close()
    
    return (rppg_signal_plot_path, rppg_spectrum_plot_path, prediction_count_path, 
            hr_trace_plot_path, hr_dist_plot_path, band_power_plot_path)

# --- rPPG multi-method and extended feature extraction ---
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
        (0.7, 2.5),
        (0.2, 0.6),
        (4.0, 8.0),
    ]
    all_features = []
    bpm_list = []
    rppg_sig_chrom = np.zeros(len(rgb_signal))
    f = np.array([])
    pxx = np.array([])
    band_power_features = []
    
    for key, rppg_sig in rppg_methods.items():
        if key == "CHROM":
            rppg_sig_chrom = rppg_sig
        if len(rppg_sig) <= 21:
            all_features.extend([0]*19)
            bpm_list.append(0)
            continue
        sig_filt = butter_bandpass_filter(rppg_sig, fs)
        f_temp, pxx_temp = periodogram(sig_filt, fs)
        valid = (f_temp >= 0.7) & (f_temp <= 4.0)
        f_temp, pxx_temp = f_temp[valid], pxx_temp[valid]
        if len(f_temp) == 0:
            all_features.extend([0]*19)
            bpm_list.append(0)
            continue
        peak_idx = np.argmax(pxx_temp)
        hr_freq, hr_bpm, hr_power = f_temp[peak_idx], f_temp[peak_idx]*60, pxx_temp[peak_idx]
        snr = hr_power / (np.sum(pxx_temp) - hr_power + 1e-8)
        band_powers = compute_band_powers(f_temp, pxx_temp, bands)
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
        bpm_list.append(hr_bpm)
        if key == "CHROM" and len(f) == 0:
            f = f_temp
            pxx = pxx_temp
            band_power_features = band_powers
    
    mean_bpm = np.mean([b for b in bpm_list if b > 0]) if bpm_list else 0
    return np.array(all_features, dtype=np.float32), mean_bpm, rppg_sig_chrom, f, pxx, band_power_features

def draw_output(frame, face_box, prediction, confidence, hr_bpm, time_stamp, track_id):
    x, y, w, h = face_box
    color = (0, 255, 0) if prediction == 'REAL' else (0, 0, 255)
    label = f"Face {track_id}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y_text = y + h + 25
    cv2.putText(frame, f"{prediction.upper()} - HR: {hr_bpm:.2f} BPM", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_text += 25
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_text += 25
    cv2.putText(frame, f"Time: {time_stamp}s", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def clamp_box(box, frame_shape):
    x, y, w, h = box
    H, W = frame_shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)

def get_video_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30
    return fps

def run_detection(video_path, video_tag, output_path=f'static/results/physio_output.mp4'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = get_video_fps(cap)
    width, height = int(cap.get(3)), int(cap.get(4))
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        test_writer = cv2.VideoWriter('test_h264.mp4', fourcc, fps, (width, height))
        if not test_writer.isOpened():
            raise RuntimeError('H.264 codec not available.')
        test_writer.release()
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    all_boxes = []
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_faces_dnn(frame)
        all_boxes.append(boxes)
        frames.append(frame)
    cap.release()
    # Track faces after collecting all boxes
    tracks = robust_track_faces(all_boxes, max_lost=5, iou_threshold=0.3, max_distance=100)
    # Prepare tracking histories
    face_rgb_history = {tid: [] for tid in tracks}
    face_pred_history = {tid: [] for tid in tracks}
    face_prob_history = {tid: [] for tid in tracks}
    face_hr_history = {tid: [] for tid in tracks}
    for frame_idx, frame in enumerate(frames):
        for track_id, detections in tracks.items():
            # Find if this face was detected in this frame
            for (fidx, box) in detections:
                if fidx == frame_idx:
                    clamped = clamp_box(box, frame.shape)
                    if clamped is None:
                        continue
                    x, y, w, h = clamped
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue
                    b, g, r = cv2.split(cv2.resize(roi, (64, 64)))
                    rgb_mean = [np.mean(r), np.mean(g), np.mean(b)]
                    face_rgb_history[track_id].append(rgb_mean)
                    rgb_window = np.array(face_rgb_history[track_id][-150:])
                    if rgb_window.shape[0] < 10:
                        continue
                    features, hr_bpm, _, _, _, _ = compute_rppg_features_multi(rgb_window, fs=fps)
                    features_scaled = scaler.transform([features])
                    prob = clf.predict_proba(features_scaled)[0][1]
                    prediction = 'FAKE' if prob > 0.4046 else 'REAL'
                    face_pred_history[track_id].append(prediction)
                    face_prob_history[track_id].append(prob)
                    face_hr_history[track_id].append(hr_bpm)
                    frame = draw_output(frame, (x, y, w, h), prediction, prob, hr_bpm, frame_idx // int(fps), track_id)
        out.write(frame)
    out.release()
    time.sleep(0.2)  # Short pause to ensure file is closed
    fixed_output_path = output_path.replace('.mp4', f'_fixed_{video_tag}.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', output_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart',
        fixed_output_path
    ])
    if os.path.exists(output_path):
        os.remove(output_path)
    output_path = fixed_output_path
    face_results = []
    MIN_PREDICTIONS = 30
    for track_id, preds in face_pred_history.items():
        if len(preds) < MIN_PREDICTIONS:
            continue
        n_real = preds.count('REAL')
        n_fake = preds.count('FAKE')
        if n_real > n_fake:
            result = 'Real'
        elif n_fake > n_real:
            result = 'Fake'
        else:
            result = 'Unknown'
        avg_hr = np.mean(face_hr_history[track_id]) if face_hr_history[track_id] else 0
        avg_conf = np.mean(face_prob_history[track_id]) if face_prob_history[track_id] else 0.0
        avg_conf = round(avg_conf * 100)
        rgb_arr = np.array(face_rgb_history[track_id])
        if rgb_arr.shape[0] >= MIN_PREDICTIONS:
            features, hr_bpm, rppg_sig, f, pxx, band_power_features = compute_rppg_features_multi(rgb_arr, fs=fps)
            plots = plot_rppg_analysis(
                rppg_sig, f, pxx, n_real, n_fake, 
                save_dir=os.path.dirname(output_path), 
                track_id=track_id,
                uid=video_tag,
                hr_history=face_hr_history[track_id],
                band_power_features=band_power_features,
                fps=fps
            )
            signal_plot, spectrum_plot, pred_count_plot, hr_trace_plot, hr_dist_plot, band_power_plot = plots
        else:
            signal_plot, spectrum_plot, pred_count_plot = None, None, None
            hr_trace_plot, hr_dist_plot, band_power_plot = None, None, None
        avg_hr = np.mean([x for x in face_hr_history[track_id] if np.isfinite(x)]) if face_hr_history[track_id] else 0
        if not np.isfinite(avg_hr):
            avg_hr = 0
        avg_hr = int(round(avg_hr))
        face_results.append({
            'track_id': track_id,
            'result': result,
            'real_count': n_real,
            'fake_count': n_fake,
            'hr': avg_hr,
            'confidence': avg_conf,
            'signal_plot': signal_plot,
            'spectrum_plot': spectrum_plot,
            'pred_count_plot': pred_count_plot,
            'hr_trace_plot': hr_trace_plot,
            'hr_dist_plot': hr_dist_plot,
            'band_power_plot': band_power_plot
        })
        # Clean up uploads folder
        if os.path.exists(video_path):
            os.remove(video_path)
    return face_results, output_path