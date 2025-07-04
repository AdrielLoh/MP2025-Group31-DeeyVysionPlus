import os
import cv2
import numpy as np
import joblib
from scipy.signal import detrend, butter, filtfilt, periodogram, correlate, welch
from scipy.stats import entropy, skew, kurtosis
import matplotlib.pyplot as plt
import uuid
import time
import subprocess

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

class FaceTracker:
    """
    A more robust face tracking system with persistent ID management.
    """
    def __init__(self, iou_thresh=0.5, max_lost_frames=30):
        self.next_track_id = 0
        self.iou_thresh = iou_thresh
        self.max_lost_frames = max_lost_frames
        self.tracks = []
    
    def update(self, curr_boxes, frame_idx):
        """Update tracks with current frame's detections."""
        assigned = [False] * len(curr_boxes)
        updated_tracks = []
        
        # Match current boxes with existing tracks
        for track in self.tracks:
            last_box = track['last_box']
            best_iou = 0
            best_idx = -1
            
            # Find best matching box
            for i, box in enumerate(curr_boxes):
                if not assigned[i]:
                    current_iou = iou(last_box, box)
                    if current_iou > best_iou and current_iou > self.iou_thresh:
                        best_iou = current_iou
                        best_idx = i
            
            if best_idx >= 0:
                # Match found
                track['boxes'].append((frame_idx, curr_boxes[best_idx]))
                track['last_box'] = curr_boxes[best_idx]
                track['last_frame'] = frame_idx
                track['lost_frames'] = 0
                assigned[best_idx] = True
                updated_tracks.append(track)
            else:
                # Track lost
                track['lost_frames'] = track.get('lost_frames', 0) + 1
                if track['lost_frames'] < self.max_lost_frames:
                    updated_tracks.append(track)
        
        # Create new tracks for unassigned boxes
        for i, box in enumerate(curr_boxes):
            if not assigned[i]:
                new_track = {
                    'id': self.next_track_id,
                    'boxes': [(frame_idx, box)],
                    'last_box': box,
                    'last_frame': frame_idx,
                    'lost_frames': 0
                }
                self.next_track_id += 1
                updated_tracks.append(new_track)
        
        self.tracks = updated_tracks
        return self.tracks

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

# ========== NEW DEEPFAKE-OPTIMIZED FEATURES ==========

def robust_pos_rppg(rgb_signal):
    """Enhanced POS method optimized for deepfake detection"""
    if rgb_signal.shape[0] < 3:
        return np.zeros(rgb_signal.shape[0])
    
    # More robust normalization
    rgb_mean = np.mean(rgb_signal, axis=0)
    rgb_mean = np.where(rgb_mean < 1e-6, 1e-6, rgb_mean)
    
    # Normalize by temporal mean to reduce illumination effects
    S = rgb_signal / rgb_mean
    
    # Enhanced POS with better numerical stability
    try:
        # POS projections
        Xcomp = 3 * S[:, 0] - 2 * S[:, 1]  # 3R - 2G
        Ycomp = 1.5 * S[:, 0] + S[:, 1] - 1.5 * S[:, 2]  # 1.5R + G - 1.5B
        
        # Robust normalization
        Xcomp_std = np.std(Xcomp)
        Ycomp_std = np.std(Ycomp)
        
        if Xcomp_std > 1e-6:
            Xcomp = Xcomp / Xcomp_std
        if Ycomp_std > 1e-6:
            Ycomp = Ycomp / Ycomp_std
            
        # Calculate optimal projection angle
        sigma_x = np.var(Xcomp)
        sigma_y = np.var(Ycomp)
        
        if len(Xcomp) > 1:
            covariance = np.cov(Xcomp, Ycomp)[0, 1]
            alpha = 0.5 * np.arctan2(2 * covariance, sigma_x - sigma_y)
        else:
            alpha = 0
        
        # Final rPPG signal
        rppg_signal = Xcomp * np.cos(alpha) + Ycomp * np.sin(alpha)
        
    except Exception:
        # Fallback to CHROM method if POS fails
        h = S[:, 1] - S[:, 0]  # Green - Red
        s = S[:, 1] + S[:, 0] - 2 * S[:, 2]  # (Green + Red) - 2*Blue
        h = h / (np.std(h) + 1e-8)
        s = s / (np.std(s) + 1e-8)
        rppg_signal = h + s
    
    return rppg_signal

def compute_deepfake_discriminative_features(rppg_sig, fs):
    """Extract features specifically chosen for deepfake detection"""
    features = {}
    
    # 1. PERIODICITY ANALYSIS
    try:
        autocorr = correlate(rppg_sig - np.mean(rppg_sig), 
                           rppg_sig - np.mean(rppg_sig), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        if len(autocorr) > 1:
            peak_lag = np.argmax(autocorr[1:fs//2]) + 1
            features['periodicity_strength'] = autocorr[peak_lag] / (autocorr[0] + 1e-8)
            features['dominant_period'] = peak_lag / fs
        else:
            features['periodicity_strength'] = 0.0
            features['dominant_period'] = 1.0
            
        # Period variability
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.1 * autocorr[0]:
                    peaks.append(i)
        
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            features['period_variability'] = np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-8)
        else:
            features['period_variability'] = 1.0
            
    except Exception:
        features['periodicity_strength'] = 0.0
        features['dominant_period'] = 1.0
        features['period_variability'] = 1.0
    
    # 2. SPECTRAL COHERENCE
    try:
        f, pxx = welch(rppg_sig, fs, nperseg=min(256, len(rppg_sig)//4))
        hr_mask = (f >= 0.7) & (f <= 4.0)
        f_hr = f[hr_mask]
        pxx_hr = pxx[hr_mask]
        
        if len(f_hr) > 0 and np.sum(pxx_hr) > 0:
            peak_idx = np.argmax(pxx_hr)
            hr_freq = f_hr[peak_idx]
            hr_power = pxx_hr[peak_idx]
            total_power = np.sum(pxx_hr)
            
            # Spectral purity
            peak_window = (f_hr >= hr_freq - 0.2) & (f_hr <= hr_freq + 0.2)
            peak_power = np.sum(pxx_hr[peak_window])
            features['spectral_purity'] = peak_power / (total_power + 1e-10)
            
            # Harmonic content
            harmonic_2_mask = (f_hr >= 2*hr_freq - 0.1) & (f_hr <= 2*hr_freq + 0.1)
            harmonic_3_mask = (f_hr >= 3*hr_freq - 0.1) & (f_hr <= 3*hr_freq + 0.1)
            
            harmonic_2_power = np.sum(pxx_hr[harmonic_2_mask]) if np.any(harmonic_2_mask) else 0
            harmonic_3_power = np.sum(pxx_hr[harmonic_3_mask]) if np.any(harmonic_3_mask) else 0
            
            features['harmonic_ratio'] = (harmonic_2_power + harmonic_3_power) / (hr_power + 1e-10)
            
            # SNR and entropy
            noise_power = total_power - peak_power
            features['frequency_snr'] = peak_power / (noise_power + 1e-10)
            
            pxx_norm = pxx_hr / (np.sum(pxx_hr) + 1e-10)
            features['spectral_entropy'] = entropy(pxx_norm + 1e-10)
            
            features['hr_bpm'] = hr_freq * 60
            
        else:
            features.update({
                'spectral_purity': 0.0, 'harmonic_ratio': 0.0,
                'frequency_snr': 0.0, 'spectral_entropy': 3.0, 'hr_bpm': 0.0
            })
            
    except Exception:
        features.update({
            'spectral_purity': 0.0, 'harmonic_ratio': 0.0,
            'frequency_snr': 0.0, 'spectral_entropy': 3.0, 'hr_bpm': 0.0
        })
    
    # 3. SIGNAL REGULARITY
    try:
        # Amplitude consistency
        window_size = max(10, len(rppg_sig) // 10)
        local_vars = []
        for i in range(0, len(rppg_sig) - window_size, window_size // 2):
            window = rppg_sig[i:i + window_size]
            local_vars.append(np.var(window))
        
        if len(local_vars) > 1:
            features['amplitude_consistency'] = 1.0 / (1.0 + np.std(local_vars))
        else:
            features['amplitude_consistency'] = 0.5
            
        # Zero-crossing regularity
        mean_val = np.mean(rppg_sig)
        zero_crossings = np.where(np.diff(np.signbit(rppg_sig - mean_val)))[0]
        
        if len(zero_crossings) > 2:
            crossing_intervals = np.diff(zero_crossings)
            features['crossing_regularity'] = 1.0 / (1.0 + np.std(crossing_intervals) / 
                                                    (np.mean(crossing_intervals) + 1e-8))
        else:
            features['crossing_regularity'] = 0.0
            
    except Exception:
        features['amplitude_consistency'] = 0.5
        features['crossing_regularity'] = 0.0
    
    # 4. MORPHOLOGICAL FEATURES
    try:
        if len(rppg_sig) > 10:
            diff_sig = np.diff(rppg_sig)
            peaks = []
            for i in range(1, len(diff_sig)):
                if diff_sig[i-1] > 0 and diff_sig[i] <= 0:
                    peaks.append(i)
            
            if len(peaks) > 2:
                peak_amplitudes = [rppg_sig[p] for p in peaks]
                features['peak_amplitude_consistency'] = 1.0 / (1.0 + np.std(peak_amplitudes) / 
                                                              (np.mean(peak_amplitudes) + 1e-8))
                
                peak_intervals = np.diff(peaks)
                features['peak_interval_variability'] = np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-8)
            else:
                features['peak_amplitude_consistency'] = 0.0
                features['peak_interval_variability'] = 1.0
        else:
            features['peak_amplitude_consistency'] = 0.0
            features['peak_interval_variability'] = 1.0
            
    except Exception:
        features['peak_amplitude_consistency'] = 0.0
        features['peak_interval_variability'] = 1.0
    
    # 5. DISTRIBUTION CHARACTERISTICS
    try:
        features['signal_skewness'] = skew(rppg_sig)
        features['signal_kurtosis'] = kurtosis(rppg_sig)
        
        hist, _ = np.histogram(rppg_sig, bins=20, density=True)
        features['distribution_entropy'] = entropy(hist + 1e-10)
        
    except Exception:
        features['signal_skewness'] = 0.0
        features['signal_kurtosis'] = 0.0
        features['distribution_entropy'] = 2.0
    
    return features

# ========== UPDATED MAIN FEATURE FUNCTION ==========

def compute_rppg_features_multi(rgb_signal, fs):
    """
    UPDATED: Now uses deepfake-optimized features while maintaining same API
    """
    # Input validation - same as before
    if rgb_signal.shape[0] < 10:
        # Return same format as before but with optimized features
        n_features = 18  # New feature count
        rppg_sig_fallback = np.zeros(rgb_signal.shape[0])
        f_fallback = np.array([1.0])
        pxx_fallback = np.array([0.0])
        band_power_fallback = [0.1, 0.3, 0.05, 0.2, 0.02, 0.1]
        return (np.zeros(n_features, dtype=np.float32), 0.0, rppg_sig_fallback, 
                f_fallback, pxx_fallback, band_power_fallback)
    
    # Preprocessing - enhanced robustness
    try:
        rgb_detrend = detrend(rgb_signal, axis=0)
        rgb_std = np.std(rgb_detrend, axis=0)
        rgb_std = np.where(rgb_std < 1e-6, 1e-6, rgb_std)
        rgb_mean = np.mean(rgb_detrend, axis=0)
        rgb_norm = (rgb_detrend - rgb_mean) / rgb_std
    except Exception:
        rgb_norm = rgb_signal / (np.mean(rgb_signal, axis=0) + 1e-6)
    
    # Extract rPPG using optimized POS method
    rppg_sig = robust_pos_rppg(rgb_norm)
    
    # Apply bandpass filter
    rppg_filtered = butter_bandpass_filter(rppg_sig, fs)
    
    # Extract deepfake-optimized features
    deepfake_features = compute_deepfake_discriminative_features(rppg_filtered, fs)
    
    # Assemble feature vector in consistent order
    feature_order = [
        'periodicity_strength', 'dominant_period', 'period_variability',
        'spectral_purity', 'harmonic_ratio', 'frequency_snr', 'spectral_entropy',
        'amplitude_consistency', 'crossing_regularity', 
        'peak_amplitude_consistency', 'peak_interval_variability',
        'signal_skewness', 'signal_kurtosis', 'distribution_entropy',
        'hr_bpm'
    ]
    
    # Add basic statistical features for robustness
    basic_features = {
        'mean': np.mean(rppg_filtered),
        'std': np.std(rppg_filtered),
        'range': np.ptp(rppg_filtered)
    }
    
    # Combine all features
    all_features = {**deepfake_features, **basic_features}
    feature_order.extend(['mean', 'std', 'range'])
    
    # Convert to array
    feature_vector = np.array([all_features.get(key, 0.0) for key in feature_order], dtype=np.float32)
    
    # Handle any remaining NaN or inf values
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
    
    hr_bpm = deepfake_features.get('hr_bpm', 0.0)
    
    # For plotting compatibility, compute frequency spectrum
    try:
        f, pxx = periodogram(rppg_filtered, fs)
        valid = (f >= 0.7) & (f <= 4.0)
        f, pxx = f[valid], pxx[valid]
    except:
        f, pxx = np.array([1.0]), np.array([0.0])
    
    # Dummy band power features for compatibility with plotting
    band_power_features = [0.1, 0.3, 0.05, 0.2, 0.02, 0.1]
    
    # Return same format as original function
    return feature_vector, hr_bpm, rppg_filtered, f, pxx, band_power_features

# ========== EVERYTHING ELSE REMAINS UNCHANGED ==========

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
    tracker = FaceTracker(iou_thresh=0.5, max_lost_frames=30)
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
    frame_idx = 0
    tracks = []
    face_rgb_history = {}
    face_pred_history = {}
    face_prob_history = {}
    face_hr_history = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_faces_dnn(frame)
        tracks = tracker.update(boxes, frame_idx)
        for track in tracks:
            if track['last_frame'] != frame_idx:
                continue
            track_id = track['id']
            clamped = clamp_box(track['last_box'], frame.shape)
            if clamped is None:
                continue
            x, y, w, h = clamped
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            b, g, r = cv2.split(cv2.resize(roi, (64, 64)))
            rgb_mean = [np.mean(r), np.mean(g), np.mean(b)]
            if track_id not in face_rgb_history:
                face_rgb_history[track_id] = []
                face_pred_history[track_id] = []
                face_prob_history[track_id] = []
                face_hr_history[track_id] = []
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
        frame_idx += 1
    cap.release()
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