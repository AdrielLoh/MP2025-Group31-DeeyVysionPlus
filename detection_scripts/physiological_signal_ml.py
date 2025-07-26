import os
import cv2
import numpy as np
import joblib
from scipy.signal import detrend, butter, filtfilt, periodogram
import scipy.stats
import matplotlib.pyplot as plt
import time
import subprocess
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import mediapipe as mp
from antropy import sample_entropy
import json

# Model and face detection setup
clf = joblib.load('models/physio_detection_xgboost_best.pkl')
scaler = joblib.load('models/physio_scaler.pkl')
FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
BASE_FEATURE_COUNT = 106
FINAL_FEATURE_COUNT = 110  # 106 + 4

def detect_faces_dnn(frame, conf_threshold=0.6):
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

def get_skin_mask_mediapipe(frame, box, face_mesh, return_landmarks=False):
    x, y, w, h = box
    x, y = max(x, 0), max(y, 0)
    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])
    face_roi = frame[y:y2, x:x2]
    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(face_roi_rgb)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks_pts = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            points = []
            for lm in face_landmarks.landmark:
                px = int(lm.x * w) + x
                py = int(lm.y * h) + y
                points.append([px, py])
            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 1)
            landmarks_pts = points  # The full set for this face
        if return_landmarks:
            return mask.astype(bool), True, landmarks_pts
        else:
            return mask.astype(bool), True
    else:
        # Fallback to bounding box
        mask[y:y2, x:x2] = 1
        if return_landmarks:
            return mask.astype(bool), False, []
        else:
            return mask.astype(bool), False
        
def align_face_by_eyes_and_nose(frame, landmarks_pts, box, output_size=(96, 112)):
    # Mediapipe landmark indices
    LEFT_EYE_IDX = 263
    RIGHT_EYE_IDX = 33
    NOSE_TIP_IDX = 1

    # Get the 2D coordinates
    left_eye = landmarks_pts[LEFT_EYE_IDX]
    right_eye = landmarks_pts[RIGHT_EYE_IDX]
    nose_tip = landmarks_pts[NOSE_TIP_IDX]

    # Target positions for aligned output
    eye_y = output_size[1] // 3
    nose_y = output_size[1] // 2
    eye_x_offset = output_size[0] // 4

    left_eye_dst = [output_size[0] - eye_x_offset, eye_y]
    right_eye_dst = [eye_x_offset, eye_y]
    nose_dst = [output_size[0] // 2, nose_y]

    # Source and destination points for affine transform
    src_pts = np.array([right_eye, left_eye, nose_tip], dtype=np.float32)
    dst_pts = np.array([right_eye_dst, left_eye_dst, nose_dst], dtype=np.float32)

    # Compute affine transform (3-point)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    aligned = cv2.warpAffine(frame, M, output_size)
    return aligned, M

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

# === rPPG multi-method and extended feature extraction ===
def safe_normalize(arr, axis=None, epsilon=1e-8):
    """Safely normalize array using NaN-aware operations"""
    std = np.nanstd(arr, axis=axis)
    return arr / (std + epsilon)

def rppg_chrom(rgb):
    """CHROM method for rPPG using NaN-aware operations"""
    try:
        if rgb.size == 0 or rgb.shape[0] < 2:
            return np.zeros(rgb.shape[0])
        
        S = rgb.copy()
        h = safe_normalize(S[:, 1] - S[:, 0])
        s = safe_normalize(S[:, 1] + S[:, 0] - 2 * S[:, 2])
        return h + s
    except Exception as e:
        return np.zeros(rgb.shape[0])

def rppg_pos(rgb):
    """POS method for rPPG using NaN-aware operations"""
    try:
        if rgb.size == 0 or rgb.shape[0] < 2:
            return np.zeros(rgb.shape[0])
        
        S = rgb.copy()
        S_mean = np.nanmean(S, axis=0)
        S_mean[S_mean == 0] = 1e-8
        S = S / S_mean
        
        Xcomp = 3 * S[:, 0] - 2 * S[:, 1]
        Ycomp = 1.5 * S[:, 0] + S[:, 1] - 1.5 * S[:, 2]
        Ycomp[Ycomp == 0] = 1e-8
        
        return Xcomp / Ycomp
    except Exception as e:
        return np.zeros(rgb.shape[0])

def rppg_green(rgb):
    """Green channel method for rPPG"""
    try:
        if rgb.size == 0:
            return np.zeros(rgb.shape[0])
        return rgb[:, 1]
    except Exception as e:
        return np.zeros(rgb.shape[0])

def compute_band_powers(f, pxx, bands):
    """Compute power in frequency bands using NaN-aware operations"""
    try:
        if len(f) == 0 or len(pxx) == 0:
            return [0] * (len(bands) * 2)
        
        total_power = np.nansum(pxx)
        if total_power == 0:
            return [0] * (len(bands) * 2)
        
        power_features = []
        for low, high in bands:
            mask = (f >= low) & (f <= high)
            power = np.nansum(pxx[mask])
            ratio = power / total_power
            power_features.extend([power, ratio])
        return power_features
    except Exception as e:
        return [0] * (len(bands) * 2)

def compute_autocorrelation(sig):
    """Compute autocorrelation using NaN-aware operations"""
    try:
        sig = sig[~np.isnan(sig)]
        if len(sig) < 2:
            return 0
        sig_centered = sig - np.nanmean(sig)
        if np.nanstd(sig_centered) == 0:
            return 0
        acf = np.correlate(sig_centered, sig_centered, mode='full')
        acf = acf[acf.size // 2:]
        if len(acf) <= 1 or acf[0] == 0:
            return 0
        peak_lag = np.nanargmax(acf[1:]) + 1
        return acf[peak_lag] / acf[0]
    except Exception as e:
        return 0

def compute_entropy(sig):
    """Compute signal entropy using NaN-aware operations"""
    try:
        sig_clean = sig[~np.isnan(sig)]
        if len(sig_clean) < 2:
            return 0
        
        hist, _ = np.histogram(sig_clean, bins=32, density=True)
        hist = hist + 1e-8
        return -np.nansum(hist * np.log2(hist))
    except Exception as e:
        return 0

def butter_bandpass_filter(signal, fs, lowcut=0.7, highcut=4.0, order=3):
    """Apply butterworth bandpass filter with NaN handling"""
    try:
        if len(signal) <= 2 * order or fs <= 0:
            return signal
        
        # Handle NaNs by interpolation or removal
        signal_clean = signal.copy()
        nan_mask = np.isnan(signal_clean)
        if np.all(nan_mask):
            return signal
        if np.any(nan_mask):
            # Simple interpolation for NaN values
            valid_indices = np.where(~nan_mask)[0]
            if len(valid_indices) < 2:
                return signal
            signal_clean[nan_mask] = np.interp(
                np.where(nan_mask)[0], 
                valid_indices, 
                signal_clean[valid_indices]
            )
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Ensure frequencies are within valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_clean)
        # Validate filtered signal before restoring NaN positions
        if np.any(np.isinf(filtered)) or np.any(np.isnan(filtered[~nan_mask])):
            return signal
        # Put NaNs back in original positions
        filtered[nan_mask] = np.nan
        
        return filtered
    except Exception as e:
        return signal

def compute_rppg_features_multi(rgb_signal, fs):
    """
    Compute rPPG features using multiple methods with simple sanity checks
    """
    try:
        f, pxx = np.array([]), np.array([])
        # NaN-aware detrending and normalization
        rgb_detrend = rgb_signal.copy()

        for i in range(rgb_signal.shape[1]):
            channel = rgb_signal[:, i]
            nan_idx = np.isnan(channel)
            if np.any(~nan_idx):
                mean_val = np.nanmean(channel)
                channel_filled = channel.copy()
                channel_filled[nan_idx] = mean_val
                detrended = detrend(channel_filled)
                detrended[nan_idx] = np.nan
                rgb_detrend[:, i] = detrended
            else:
                rgb_detrend[:, i] = np.nan
        rgb_norm = rgb_detrend.copy()
        for i in range(rgb_norm.shape[1]):
            std = np.nanstd(rgb_norm[:, i])
            if std > 0:
                rgb_norm[:, i] = (rgb_norm[:, i] - np.nanmean(rgb_norm[:, i])) / std

        # Apply rPPG methods
        rppg_methods = {
            "CHROM": rppg_chrom(rgb_norm),
            "POS": rppg_pos(rgb_norm),
            "GREEN": rppg_green(rgb_norm),
        }
        method_names = ["CHROM", "POS", "GREEN"]

        bands = [(0.7, 2.5), (0.2, 0.6), (4.0, 8.0)]

        # Feature collection
        all_features = []
        rppg_signals = []
        mean_bpm = 0
        method_hr_bpm_values = []  # For sanity checking
        method_snr_values = []     # For sanity checking
        
        # Per-method feature extraction
        for method in method_names:
            rppg_sig = rppg_methods[method]
            rppg_signals.append(rppg_sig)
            if len(rppg_sig) <= 21 or np.all(np.isnan(rppg_sig)):
                all_features.extend([0]*29)
                method_hr_bpm_values.append(0)
                method_snr_values.append(0)
                continue

            # Bandpass filter
            sig_filt = butter_bandpass_filter(rppg_sig, fs)
            sig_clean = sig_filt[~np.isnan(sig_filt)]

            # Basic stats
            mean = np.nanmean(sig_filt)
            std = np.nanstd(sig_filt)
            maxval = np.nanmax(sig_filt)
            minval = np.nanmin(sig_filt)
            ptp = maxval - minval

            # Frequency stats
            try:
                if len(sig_clean) > 10:
                    f, pxx = periodogram(sig_clean, fs)
                    valid = (f >= 0.7) & (f <= 4.0)
                    f, pxx = f[valid], pxx[valid]
                else:
                    f, pxx = np.array([]), np.array([])
                if len(f) == 0 or np.all(np.isnan(pxx)):
                    hr_freq = 0
                    hr_bpm = 0
                    hr_power = 0
                    snr = 0
                    band_powers = [0]*6
                else:
                    peak_idx = np.nanargmax(pxx)
                    hr_freq = f[peak_idx]
                    hr_bpm = hr_freq * 60
                    hr_power = pxx[peak_idx]
                    total_power = np.nansum(pxx)
                    snr = hr_power / (total_power - hr_power + 1e-8) if total_power > hr_power else 0
                    band_powers = compute_band_powers(f, pxx, bands)
            except Exception:
                hr_freq = hr_bpm = hr_power = snr = 0
                band_powers = [0]*6

            # Store for sanity checking
            method_hr_bpm_values.append(hr_bpm)
            method_snr_values.append(snr)

            # Signal shape/stats
            ac_peak = compute_autocorrelation(sig_filt)
            ent = compute_entropy(sig_filt)
            try:
                kurt = scipy.stats.kurtosis(sig_filt, nan_policy='omit')
                skewness = scipy.stats.skew(sig_filt, nan_policy='omit')
            except Exception:
                kurt = skewness = 0
            rms = np.sqrt(np.nanmean(sig_filt**2))
            coeff_var = std / (mean + 1e-8) if abs(mean) > 1e-8 else 0
            tkeo = np.nanmean(np.square(sig_filt[1:-1]) - sig_filt[:-2]*sig_filt[2:]) if len(sig_filt) > 2 else 0
            
            # Sample entropy and spectral flatness
            try:
                samp_ent = sample_entropy(sig_filt)
                if len(sig_clean) > 10:
                    _, psd = periodogram(sig_clean, fs)
                    if len(psd) > 0 and np.all(psd > 0):
                        geometric_mean = np.exp(np.mean(np.log(psd + 1e-12)))
                        arithmetic_mean = np.mean(psd)
                        spec_flat = geometric_mean / (arithmetic_mean + 1e-12)
                    else:
                        spec_flat = 0
                else:
                    spec_flat = 0
            except Exception:
                samp_ent = 0
                spec_flat = 0

            # Peak-to-noise ratio
            peak2noise = (np.nanmax(np.abs(sig_filt)) / (np.nanstd(sig_filt) + 1e-8)) if np.nanstd(sig_filt) > 0 else 0
            root_diff = np.nanmean(np.abs(np.diff(sig_filt)))
            perc25 = np.nanpercentile(sig_filt, 25)
            perc75 = np.nanpercentile(sig_filt, 75)
            hr_valid = int(hr_bpm > 30 and hr_bpm < 180 and hr_power > 0 and not np.isnan(hr_bpm))

            # Append features for this method
            method_features = [
                mean, std, maxval, minval, ptp,
                hr_freq, hr_bpm, hr_power, snr,
                ac_peak, ent, kurt, skewness, rms, coeff_var, tkeo, samp_ent, spec_flat,
                band_powers[0], band_powers[1], band_powers[2], band_powers[3], band_powers[4], band_powers[5],
                peak2noise, root_diff, perc25, perc75, hr_valid
            ]
            all_features.extend(method_features)
            mean_bpm += hr_bpm if hr_valid else 0

        # Inter-method correlations (6 features)
        corr_features = []
        signals_for_corr = []
        for method in method_names:
            rppg_sig = rppg_methods[method]
            sig_filt = butter_bandpass_filter(rppg_sig, fs)
            signals_for_corr.append(sig_filt)

        def nan_corr(a, b, method="pearson"):
            valid = ~np.isnan(a) & ~np.isnan(b)
            if np.sum(valid) < 8:
                return 0
            if method == "pearson":
                return np.corrcoef(a[valid], b[valid])[0,1]
            elif method == "spearman":
                return scipy.stats.spearmanr(a[valid], b[valid])[0]
            return 0

        corr_features.append(nan_corr(signals_for_corr[0], signals_for_corr[2], "pearson"))
        corr_features.append(nan_corr(signals_for_corr[0], signals_for_corr[1], "pearson"))
        corr_features.append(nan_corr(signals_for_corr[1], signals_for_corr[2], "pearson"))
        corr_features.append(nan_corr(signals_for_corr[0], signals_for_corr[2], "spearman"))
        corr_features.append(nan_corr(signals_for_corr[0], signals_for_corr[1], "spearman"))
        corr_features.append(nan_corr(signals_for_corr[1], signals_for_corr[2], "spearman"))

        all_features.extend(corr_features)

        # === Temporal/Aggregated Features ===
        hr_bpm_arr = []
        snr_arr = []
        band_ratio_arr = []
        hr_valid_arr = []
        
        for method_idx in range(3):
            start_idx = method_idx * 29
            if start_idx + 28 < len(all_features):
                hr_bpm_arr.append(all_features[start_idx + 6])
                snr_arr.append(all_features[start_idx + 8])
                band_ratio_arr.append(all_features[start_idx + 19])
                hr_valid_arr.append(all_features[start_idx + 28])
            else:
                hr_bpm_arr.extend([np.nan])
                snr_arr.extend([np.nan])
                band_ratio_arr.extend([np.nan])
                hr_valid_arr.extend([0])
        
        hr_bpm_arr = np.array(hr_bpm_arr)
        snr_arr = np.array(snr_arr)
        band_ratio_arr = np.array(band_ratio_arr)
        hr_valid_arr = np.array(hr_valid_arr)

        # Compute dropouts
        hr_valid_bool = hr_valid_arr > 0
        num_dropouts = 0
        if np.any(~np.isnan(hr_bpm_arr)):
            prev = hr_valid_bool[0]
            for v in hr_valid_bool[1:]:
                if prev and not v:
                    num_dropouts += 1
                prev = v
        else:
            num_dropouts = 0

        prop_valid_hr = np.sum(hr_valid_bool) / len(hr_valid_bool) if len(hr_valid_bool) > 0 else 0

        # Temporal statistics
        hr_std = np.nanstd(hr_bpm_arr)
        hr_delta = np.nanmean(np.abs(np.diff(hr_bpm_arr)))
        hr_min = np.nanmin(hr_bpm_arr)
        hr_max = np.nanmax(hr_bpm_arr)
        snr_std = np.nanstd(snr_arr)
        snr_min = np.nanmin(snr_arr)
        snr_max = np.nanmax(snr_arr)
        band_ratio_std = np.nanstd(band_ratio_arr)
        band_ratio_mean = np.nanmean(band_ratio_arr)

        temporal_feats = [
            hr_std, hr_delta, hr_min, hr_max, num_dropouts, prop_valid_hr,
            snr_std, snr_min, snr_max, band_ratio_std, band_ratio_mean
        ]

        all_features.extend(temporal_feats)

        # Additional features
        prop_nan_window = np.isnan(rgb_signal).sum() / np.prod(rgb_signal.shape)
        prop_low_snr_methods = np.mean(snr_arr < 1)
        all_features.extend([prop_nan_window, prop_low_snr_methods])

        # Convert to numpy array
        features_array = np.array(all_features, dtype=np.float32)
        
        return (
            features_array, 
            mean_bpm/3, 
            rppg_methods["CHROM"], 
            f if 'f' in locals() else np.array([]), 
            pxx if 'pxx' in locals() else np.array([])
        )

    except Exception as e:
        return np.zeros(BASE_FEATURE_COUNT, dtype=np.float32), 0, np.zeros(rgb_signal.shape[0]), np.array([]), np.array([])
    
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

def run_detection(video_path, video_tag, output_dir):
    # === Check for cache results ===
    result_path = os.path.join(output_dir, "cached_results.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            cached = json.load(f)
        face_results = cached["face_results"]
        output_path = cached.get("output_path")
        return face_results, output_path
    
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )
    
    output_path = os.path.join(output_dir, f'physio_output.mp4')
    os.makedirs(output_dir, exist_ok=True)

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
        # 1. Letterbox the frame
        padded_frame, scale, (pad_left, pad_top) = letterbox_resize(frame, target_size=(640, 360), pad_color=0)
        boxes = detect_faces_dnn(padded_frame)
        # 2. Map detected boxes back to original frame coordinates
        orig_boxes = []
        for x, y, w, h in boxes:
            # Remove padding, scale back to original
            x_orig = int((x - pad_left) / scale)
            y_orig = int((y - pad_top) / scale)
            w_orig = int(w / scale)
            h_orig = int(h / scale)
            orig_boxes.append((x_orig, y_orig, w_orig, h_orig))
        all_boxes.append(orig_boxes)
        frames.append(frame)  # keep original frame for overlay/output

    cap.release()
    # Track faces after collecting all boxes
    tracks = robust_track_faces(all_boxes, max_lost=5, iou_threshold=0.3, max_distance=100)
    # Prepare tracking histories
    face_rgb_history = {tid: [] for tid in tracks}
    face_pred_history = {tid: [] for tid in tracks}
    face_prob_history = {tid: [] for tid in tracks}
    face_hr_history = {tid: [] for tid in tracks}
    face_area_history = {tid: [] for tid in tracks}
    for frame_idx, frame in enumerate(frames):
        for track_id, detections in tracks.items():
            # Find if this face was detected in this frame
            for (fidx, box) in detections:
                if fidx == frame_idx:
                    clamped = clamp_box(box, frame.shape)
                    if clamped is None:
                        continue
                    x, y, w, h = clamped
                    # 1. Get convex hull mask for face
                    mask, used_hull, landmarks_pts = get_skin_mask_mediapipe(frame, (x, y, w, h), face_mesh, return_landmarks=True)
                    if mask.sum() == 0 or len(landmarks_pts) < 468:
                        continue
                    face_area_history[track_id].append(mask.sum())

                    # 2. Extract RGB means from aligned face, or fallback to unaligned if alignment fails
                    try:
                        aligned_face, M = align_face_by_eyes_and_nose(frame, landmarks_pts, (x, y, w, h))
                        aligned_mask = cv2.warpAffine(mask.astype(np.uint8), M, (aligned_face.shape[1], aligned_face.shape[0])) > 0
                        b, g, r = cv2.split(aligned_face)
                        r_mean = np.mean(r[aligned_mask])
                        g_mean = np.mean(g[aligned_mask])
                        b_mean = np.mean(b[aligned_mask])
                    except Exception as e:
                        # fallback to unaligned mask
                        roi = frame[y:y+h, x:x+w]
                        local_mask = mask[y:y+h, x:x+w]
                        if roi.size == 0 or local_mask.sum() == 0:
                            continue
                        b, g, r = cv2.split(roi)
                        r_mean = np.mean(r[local_mask])
                        g_mean = np.mean(g[local_mask])
                        b_mean = np.mean(b[local_mask])
                    rgb_mean = [r_mean, g_mean, b_mean]

                    face_rgb_history[track_id].append(rgb_mean)

                    rgb_window = np.array(face_rgb_history[track_id][-150:])
                    features, hr_bpm, _, _, _ = compute_rppg_features_multi(rgb_window, fs=fps)

                    if features.shape[0] != FINAL_FEATURE_COUNT:
                        # Simple safety check
                        print(f"Warning: feature count mismatch: {features.shape[0]} != {FINAL_FEATURE_COUNT}")
                    features_scaled = scaler.transform([features])
                    prob = clf.predict_proba(features_scaled)[0][1]

                    prediction = 'FAKE' if prob > 0.4923 else 'REAL'
                    face_pred_history[track_id].append(prediction)
                    face_prob_history[track_id].append(prob)
                    face_hr_history[track_id].append(hr_bpm)
                    frame = draw_output(frame, (x, y, w, h), prediction, prob, hr_bpm, frame_idx // int(fps), track_id)
        out.write(frame)
    out.release()
    time.sleep(0.2)  # Short pause to ensure file is closed
    fixed_output_path = output_path.replace('.mp4', f'_fixed.mp4')
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
            window_length_feat = rgb_arr.shape[0]
            valid_ratio = np.sum(~np.isnan(rgb_arr[:, 0])) / 150
            area_arr = np.array(face_area_history[track_id])
            if area_arr.size == 0:
                avg_face_area = 0
                med_face_area = 0
            else:
                avg_face_area = float(np.mean(area_arr))
                med_face_area = float(np.median(area_arr))
            features, hr_bpm, rppg_sig, f, pxx = compute_rppg_features_multi(rgb_arr, fs=fps)
            features = np.concatenate([features, [valid_ratio, avg_face_area, med_face_area, window_length_feat]])
            plots = plot_rppg_analysis(
                rppg_sig, f, pxx, n_real, n_fake, 
                save_dir=output_dir, 
                track_id=track_id,
                uid=video_tag,
                hr_history=face_hr_history[track_id],
                # band_power_features=band_power_features,
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

        # ===== Cache results for future use to reduce computing =====
        result_dict = {
            "face_results": face_results,
            "output_path": output_path
        }
        with open(result_path, "w") as f:
            json.dump(result_dict, f)

    return face_results, output_path
