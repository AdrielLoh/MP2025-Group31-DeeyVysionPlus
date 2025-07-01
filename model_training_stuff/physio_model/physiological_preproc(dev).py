import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy.signal import detrend, butter, filtfilt, periodogram, correlate, welch
from scipy.stats import entropy, skew, kurtosis
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Augmentation Config ---
AUGMENTATION = {
    "enabled": True,
    # Probabilities
    "p_brightness": 0.25,
    "p_contrast": 0.25,
    "p_noise": 0.25,
    "p_color_shift": 0.25,
    "p_motion_blur": 0.25,
    "p_jpeg": 0.25,
    "p_gamma": 0.25,
    "p_occlusion": 0.1,
    "p_crop": 0.05,
    # Strengths
    "brightness": 0.10,
    "contrast": 0.10,
    "gaussian_noise": 5,
    "color_shift": 10,
    "motion_blur_max_ksize": 5,
    "jpeg_min_quality": 60,
    "jpeg_max_quality": 95,
    "gamma_min": 0.7,
    "gamma_max": 1.3,
    "occlusion_max_frac": 0.18,
    "crop_max_frac": 0.13,
}

# --- Augmentation methods ---
def safe_clip(arr, min_val=0, max_val=255):
    """Safely clip array values"""
    return np.clip(arr, min_val, max_val)

def augment_brightness(roi, cfg):
    """Apply brightness augmentation"""
    try:
        factor = 1.0 + np.random.uniform(-cfg["brightness"], cfg["brightness"])
        return safe_clip(roi * factor)
    except Exception as e:
        logger.warning(f"Brightness augmentation failed: {e}")
        return roi

def augment_contrast(roi, cfg):
    """Apply contrast augmentation"""
    try:
        if roi.size == 0:
            return roi
        mean = np.mean(roi, axis=(0, 1), keepdims=True)
        factor = 1.0 + np.random.uniform(-cfg["contrast"], cfg["contrast"])
        return safe_clip((roi - mean) * factor + mean)
    except Exception as e:
        logger.warning(f"Contrast augmentation failed: {e}")
        return roi

def augment_gaussian_noise(roi, cfg):
    """Apply Gaussian noise augmentation"""
    try:
        noise = np.random.normal(0, cfg["gaussian_noise"], roi.shape)
        return safe_clip(roi + noise)
    except Exception as e:
        logger.warning(f"Noise augmentation failed: {e}")
        return roi

def augment_color_shift(roi, cfg):
    """Apply color shift augmentation"""
    try:
        if len(roi.shape) != 3 or roi.shape[2] != 3:
            return roi
        shift = np.random.randint(-cfg["color_shift"], cfg["color_shift"] + 1, size=(1, 1, 3))
        return safe_clip(roi + shift)
    except Exception as e:
        logger.warning(f"Color shift augmentation failed: {e}")
        return roi

def augment_motion_blur(roi, cfg):
    """Apply motion blur augmentation"""
    try:
        if roi.size == 0:
            return roi
        
        ksize = 3 if cfg["motion_blur_max_ksize"] < 5 else np.random.choice([3, 5])
        
        # Create kernel
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if np.random.rand() < 0.5:
            # Horizontal blur
            kernel[ksize//2, :] = 1.0
        else:
            # Vertical blur
            kernel[:, ksize//2] = 1.0
        kernel /= ksize
        
        # Apply blur
        roi_uint8 = safe_clip(roi).astype(np.uint8)
        blurred = cv2.filter2D(roi_uint8, -1, kernel)
        return blurred.astype(np.float64)
    except Exception as e:
        logger.warning(f"Motion blur augmentation failed: {e}")
        return roi

def augment_jpeg_compression(roi, cfg):
    """Apply JPEG compression augmentation"""
    try:
        if roi.size == 0:
            return roi
        
        roi_uint8 = safe_clip(roi).astype(np.uint8)
        quality = np.random.randint(cfg["jpeg_min_quality"], cfg["jpeg_max_quality"] + 1)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        result, encimg = cv2.imencode('.jpg', roi_uint8, encode_param)
        if result and encimg is not None:
            decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            if decoded is not None:
                return decoded.astype(np.float64)
        return roi
    except Exception as e:
        logger.warning(f"JPEG compression augmentation failed: {e}")
        return roi

def augment_gamma(roi, cfg):
    """Apply gamma correction augmentation"""
    try:
        if roi.size == 0:
            return roi
        
        gamma = np.random.uniform(cfg["gamma_min"], cfg["gamma_max"])
        roi_norm = safe_clip(roi / 255.0, 0, 1)
        roi_gamma = np.power(roi_norm, gamma)
        return safe_clip(roi_gamma * 255.0)
    except Exception as e:
        logger.warning(f"Gamma augmentation failed: {e}")
        return roi

def augment_occlusion(roi, cfg):
    """Apply occlusion augmentation"""
    try:
        if roi.size == 0 or len(roi.shape) != 3:
            return roi
        
        h, w, c = roi.shape
        if h <= 10 or w <= 10:
            return roi
        
        occ_h = max(1, int(h * np.random.uniform(0.08, cfg["occlusion_max_frac"])))
        occ_w = max(1, int(w * np.random.uniform(0.08, cfg["occlusion_max_frac"])))
        
        y1 = np.random.randint(0, max(1, h - occ_h))
        x1 = np.random.randint(0, max(1, w - occ_w))
        
        color = np.mean(roi, axis=(0, 1)) + np.random.randint(-10, 11, size=(c,))
        color = safe_clip(color)
        
        roi_copy = roi.copy()
        roi_copy[y1:y1+occ_h, x1:x1+occ_w] = color
        return roi_copy
    except Exception as e:
        logger.warning(f"Occlusion augmentation failed: {e}")
        return roi

def augment_random_crop(roi, cfg):
    """Apply random crop augmentation"""
    try:
        if roi.size == 0 or len(roi.shape) != 3:
            return roi
        
        h, w, c = roi.shape
        if h <= 20 or w <= 20:
            return roi
        
        crop_frac = np.random.uniform(0, cfg["crop_max_frac"])
        ch = int(h * crop_frac)
        cw = int(w * crop_frac)
        
        if ch >= h or cw >= w:
            return roi
        
        y1 = np.random.randint(0, ch + 1)
        x1 = np.random.randint(0, cw + 1)
        y2 = h - (ch - y1)
        x2 = w - (cw - x1)
        
        if y2 - y1 < 10 or x2 - x1 < 10:
            return roi
        
        cropped = roi[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized
    except Exception as e:
        logger.warning(f"Random crop augmentation failed: {e}")
        return roi

def augment_frame(roi, aug_cfg):
    """Apply random augmentations to frame"""
    if not aug_cfg or not aug_cfg.get("enabled", False) or roi.size == 0:
        return safe_clip(roi).astype(np.uint8)
    
    try:
        # Convert to float for processing
        roi = roi.astype(np.float64)
        
        # Apply augmentations randomly
        augmentations = []
        
        if np.random.rand() < aug_cfg["p_brightness"]:
            augmentations.append(lambda x: augment_brightness(x, aug_cfg))
        if np.random.rand() < aug_cfg["p_contrast"]:
            augmentations.append(lambda x: augment_contrast(x, aug_cfg))
        if np.random.rand() < aug_cfg["p_noise"]:
            augmentations.append(lambda x: augment_gaussian_noise(x, aug_cfg))
        if np.random.rand() < aug_cfg["p_color_shift"]:
            augmentations.append(lambda x: augment_color_shift(x, aug_cfg))
        if np.random.rand() < aug_cfg["p_motion_blur"]:
            augmentations.append(lambda x: augment_motion_blur(x, aug_cfg))
        if np.random.rand() < aug_cfg["p_jpeg"]:
            augmentations.append(lambda x: augment_jpeg_compression(x, aug_cfg))
        if np.random.rand() < aug_cfg["p_gamma"]:
            augmentations.append(lambda x: augment_gamma(x, aug_cfg))
        if np.random.rand() < aug_cfg["p_occlusion"]:
            augmentations.append(lambda x: augment_occlusion(x, aug_cfg))
        if np.random.rand() < aug_cfg["p_crop"]:
            augmentations.append(lambda x: augment_random_crop(x, aug_cfg))
        
        # Shuffle and apply augmentations
        np.random.shuffle(augmentations)
        for func in augmentations:
            roi = func(roi)
        
        return safe_clip(roi).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Frame augmentation failed: {e}")
        return safe_clip(roi).astype(np.uint8)

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

def butter_bandpass_filter(signal, fs, lowcut=0.7, highcut=4.0, order=3):
    """Apply butterworth bandpass filter"""
    try:
        if len(signal) <= 2 * order or fs <= 0:
            return signal
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure frequencies are within valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = butter(order, [low, high], btype='band')
        
        # Check if signal is long enough for filtfilt
        padlen = 3 * max(len(a), len(b))
        if len(signal) <= padlen:
            return signal
            
        return filtfilt(b, a, signal)
    except Exception as e:
        logger.warning(f"Bandpass filter failed: {e}")
        return signal

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
    
    # Return same format as original function - note the 5th return value for compatibility
    return feature_vector, hr_bpm, rppg_filtered, f, pxx

# ========== REST OF THE CODE REMAINS UNCHANGED ==========

def get_face_net():
    """Initialize face detection network"""
    try:
        # Update these paths according to your setup
        FACE_PROTO = 'models/weights-prototxt.txt'
        FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'
        
        if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
            logger.error(f"Face detection model files not found. Please update paths.")
            return None
        
        face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        return face_net
    except Exception as e:
        logger.error(f"Failed to load face detection model: {e}")
        return None

def detect_faces_dnn(frame, face_net, conf_threshold=0.5):
    """Detect faces using DNN"""
    try:
        if face_net is None or frame is None or frame.size == 0:
            return []
        
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return []
        
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
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return []

def iou(boxA, boxB):
    """Compute intersection over union of two boxes"""
    try:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, boxA[2] * boxA[3])
        boxBArea = max(1, boxB[2] * boxB[3])
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    except:
        return 0

def compute_box_distance(box1, box2):
    """Compute Euclidean distance between box centers"""
    try:
        center1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
        center2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    except:
        return float('inf')

def compute_box_similarity(box1, box2):
    """Compute similarity between boxes using multiple metrics"""
    try:
        # IoU score
        iou_score = iou(box1, box2)
        
        # Size similarity (ratio of areas)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        size_ratio = min(area1, area2) / (max(area1, area2) + 1e-8)
        
        # Distance penalty
        distance = compute_box_distance(box1, box2)
        max_dim = max(box1[2], box1[3], box2[2], box2[3])
        distance_score = max(0, 1 - distance / (max_dim * 2))
        
        # Combined score with weights
        combined_score = 0.6 * iou_score + 0.2 * size_ratio + 0.2 * distance_score
        
        return combined_score
    except:
        return 0

class RobustFaceTracker:
    """
    Enhanced face tracking with collision prevention and data leakage protection
    """
    def __init__(self, 
                 similarity_thresh=0.4,
                 max_lost_frames=10,
                 min_track_length=5,
                 min_track_confidence=0.6,
                 overlap_penalty=0.5,
                 max_tracks_per_frame=5):
        """
        Args:
            similarity_thresh: Minimum similarity score to match boxes
            max_lost_frames: Maximum frames a track can be lost before termination
            min_track_length: Minimum number of detections for a valid track
            min_track_confidence: Minimum average confidence for a valid track
            overlap_penalty: Penalty factor for overlapping tracks
            max_tracks_per_frame: Maximum simultaneous tracks allowed
        """
        self.similarity_thresh = similarity_thresh
        self.max_lost_frames = max_lost_frames
        self.min_track_length = min_track_length
        self.min_track_confidence = min_track_confidence
        self.overlap_penalty = overlap_penalty
        self.max_tracks_per_frame = max_tracks_per_frame
        self.next_track_id = 0
        self.tracks = []
        self.terminated_tracks = []
    
    def _compute_track_overlap(self, track1, track2, frame_idx):
        """Compute overlap between two tracks at a specific frame"""
        box1 = None
        box2 = None
        
        # Find boxes at frame_idx
        for fid, box in track1['boxes']:
            if fid == frame_idx:
                box1 = box
                break
        
        for fid, box in track2['boxes']:
            if fid == frame_idx:
                box2 = box
                break
        
        if box1 is not None and box2 is not None:
            return iou(box1, box2)
        return 0
    
    def _resolve_track_conflicts(self, tracks, frame_idx):
        """Resolve conflicts between overlapping tracks"""
        if len(tracks) <= 1:
            return tracks
        
        # Compute pairwise overlaps
        overlap_matrix = np.zeros((len(tracks), len(tracks)))
        for i in range(len(tracks)):
            for j in range(i+1, len(tracks)):
                overlap = self._compute_track_overlap(tracks[i], tracks[j], frame_idx)
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap
        
        # Identify conflicting tracks
        conflicts = []
        for i in range(len(tracks)):
            for j in range(i+1, len(tracks)):
                if overlap_matrix[i, j] > 0.3:  # Significant overlap threshold
                    conflicts.append((i, j, overlap_matrix[i, j]))
        
        # Resolve conflicts by keeping higher confidence tracks
        tracks_to_remove = set()
        for i, j, overlap in sorted(conflicts, key=lambda x: x[2], reverse=True):
            if i in tracks_to_remove or j in tracks_to_remove:
                continue
            
            # Compare track quality
            conf_i = tracks[i].get('avg_confidence', 0)
            conf_j = tracks[j].get('avg_confidence', 0)
            len_i = len(tracks[i]['boxes'])
            len_j = len(tracks[j]['boxes'])
            
            # Quality score: combination of confidence and length
            quality_i = conf_i * np.log1p(len_i)
            quality_j = conf_j * np.log1p(len_j)
            
            if quality_i < quality_j:
                tracks_to_remove.add(i)
            else:
                tracks_to_remove.add(j)
        
        # Return non-conflicting tracks
        return [track for i, track in enumerate(tracks) if i not in tracks_to_remove]
    
    def _prune_tracks(self, frame_idx):
        """Remove inactive or low-quality tracks"""
        active_tracks = []
        
        for track in self.tracks:
            # Check if track is too old
            if frame_idx - track['last_frame'] > self.max_lost_frames:
                track['terminated_frame'] = frame_idx
                self.terminated_tracks.append(track)
                continue
            
            # Check if track has sufficient detections
            if (frame_idx - track['start_frame'] > self.min_track_length * 2 and 
                len(track['boxes']) < self.min_track_length):
                track['terminated_frame'] = frame_idx
                self.terminated_tracks.append(track)
                continue
            
            active_tracks.append(track)
        
        # Limit number of simultaneous tracks
        if len(active_tracks) > self.max_tracks_per_frame:
            # Sort by quality (confidence * length)
            active_tracks.sort(
                key=lambda t: t.get('avg_confidence', 0) * len(t['boxes']), 
                reverse=True
            )
            
            # Move excess tracks to terminated
            for track in active_tracks[self.max_tracks_per_frame:]:
                track['terminated_frame'] = frame_idx
                self.terminated_tracks.append(track)
            
            active_tracks = active_tracks[:self.max_tracks_per_frame]
        
        self.tracks = active_tracks
    
    def update(self, boxes, frame_idx, confidences=None):
        """Update tracks with new detections"""
        if confidences is None:
            confidences = [1.0] * len(boxes)
        
        # Prune old/bad tracks
        self._prune_tracks(frame_idx)
        
        # Match boxes to existing tracks
        assigned_boxes = [False] * len(boxes)
        assigned_tracks = [False] * len(self.tracks)
        
        # Create cost matrix for Hungarian algorithm
        if len(self.tracks) > 0 and len(boxes) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(boxes)))
            
            for i, track in enumerate(self.tracks):
                for j, box in enumerate(boxes):
                    similarity = compute_box_similarity(track['last_box'], box)
                    # Convert similarity to cost (higher similarity = lower cost)
                    cost_matrix[i, j] = 1 - similarity
            
            # Solve assignment problem
            from scipy.optimize import linear_sum_assignment
            track_indices, box_indices = linear_sum_assignment(cost_matrix)
            
            # Apply assignments if similarity threshold is met
            for track_idx, box_idx in zip(track_indices, box_indices):
                similarity = 1 - cost_matrix[track_idx, box_idx]
                if similarity >= self.similarity_thresh:
                    track = self.tracks[track_idx]
                    
                    # Update track
                    track['boxes'].append((frame_idx, boxes[box_idx]))
                    track['last_box'] = boxes[box_idx]
                    track['last_frame'] = frame_idx
                    track['lost_frames'] = 0
                    
                    # Update confidence
                    track['confidences'].append(confidences[box_idx])
                    track['avg_confidence'] = np.mean(track['confidences'])
                    
                    assigned_boxes[box_idx] = True
                    assigned_tracks[track_idx] = True
        
        # Update lost frame counters for unassigned tracks
        for i, track in enumerate(self.tracks):
            if not assigned_tracks[i]:
                track['lost_frames'] += 1
        
        # Create new tracks for unassigned boxes
        for i, box in enumerate(boxes):
            if not assigned_boxes[i]:
                new_track = {
                    'id': self.next_track_id,
                    'boxes': [(frame_idx, box)],
                    'last_box': box,
                    'start_frame': frame_idx,
                    'last_frame': frame_idx,
                    'lost_frames': 0,
                    'confidences': [confidences[i]],
                    'avg_confidence': confidences[i]
                }
                self.next_track_id += 1
                self.tracks.append(new_track)
        
        # Resolve conflicts between tracks
        self.tracks = self._resolve_track_conflicts(self.tracks, frame_idx)
        
        return self.tracks
    
    def get_valid_tracks(self, min_gap_frames=30):
        """
        Get all valid tracks with gap enforcement to prevent data leakage
        
        Args:
            min_gap_frames: Minimum frame gap between tracks to prevent overlap
        """
        all_tracks = self.tracks + self.terminated_tracks
        
        # Filter by minimum length and confidence
        valid_tracks = []
        for track in all_tracks:
            if (len(track['boxes']) >= self.min_track_length and
                track.get('avg_confidence', 0) >= self.min_track_confidence):
                valid_tracks.append(track)
        
        # Sort tracks by start frame
        valid_tracks.sort(key=lambda t: t['start_frame'])
        
        # Apply gap enforcement
        final_tracks = []
        last_end_frame = -min_gap_frames
        
        for track in valid_tracks:
            track_start = track['start_frame']
            track_end = track['last_frame']
            
            # Check if sufficient gap exists
            if track_start >= last_end_frame + min_gap_frames:
                final_tracks.append(track)
                last_end_frame = track_end
        
        # Convert to tracklets format
        tracklets = {
            track['id']: track['boxes'] 
            for track in final_tracks
        }
        
        return tracklets

def track_faces(all_face_boxes, iou_thresh=0.5):
    """
    Enhanced face tracking with robust collision handling
    """
    try:
        tracker = RobustFaceTracker(
            similarity_thresh=0.4,
            max_lost_frames=10,
            min_track_length=5,
            min_track_confidence=0.6,
            overlap_penalty=0.5,
            max_tracks_per_frame=5
        )
        
        # Update tracker with all frames
        for frame_idx, boxes in enumerate(all_face_boxes):
            if boxes:
                # Generate dummy confidences if not available
                confidences = [0.9] * len(boxes)
                tracker.update(boxes, frame_idx, confidences)
        
        # Get valid tracks with gap enforcement
        tracklets = tracker.get_valid_tracks(min_gap_frames=30)
        
        logger.info(f"Tracking complete: {len(tracklets)} valid tracks from {len(all_face_boxes)} frames")
        
        return tracklets
        
    except Exception as e:
        logger.error(f"Enhanced face tracking failed: {e}")
        return {}

def extract_rgb_signal_track(frames, face_boxes, augmentation_cfg=None):
    """Extract RGB signal from tracked faces"""
    try:
        r_means, g_means, b_means, face_mask = [], [], [], []
        
        for frame, box in zip(frames, face_boxes):
            if box is not None and frame is not None and frame.size > 0:
                x, y, w, h = box
                x, y = max(x, 0), max(y, 0)
                x2 = min(x + w, frame.shape[1])
                y2 = min(y + h, frame.shape[0])
                
                if x2 > x and y2 > y:
                    roi = frame[y:y2, x:x2]
                    
                    if augmentation_cfg and augmentation_cfg.get("enabled", False):
                        roi = augment_frame(roi, augmentation_cfg)
                    
                    if roi.size > 0:
                        # Ensure ROI is 3-channel
                        if len(roi.shape) == 3 and roi.shape[2] == 3:
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
        
        rgb_signal = np.stack([r_means, g_means, b_means], axis=-1)
        return rgb_signal, np.array(face_mask)
    except Exception as e:
        logger.warning(f"RGB signal extraction failed: {e}")
        return np.zeros((len(frames), 3)), np.zeros(len(frames))

def sliding_windows_with_mask(signal, mask, window_size, hop_size, min_face_ratio=0.8):
    """Create sliding windows with face mask filtering"""
    try:
        if signal.size == 0 or len(signal) < window_size:
            return np.empty((0, window_size, signal.shape[1]))
        
        windows = []
        for start in range(0, len(signal) - window_size + 1, hop_size):
            end = start + window_size
            window_mask = mask[start:end]
            if np.mean(window_mask) >= min_face_ratio:
                windows.append(signal[start:end, :])
        
        return np.stack(windows) if windows else np.empty((0, window_size, signal.shape[1]))
    except Exception as e:
        logger.warning(f"Sliding window creation failed: {e}")
        return np.empty((0, window_size, signal.shape[1] if signal.size > 0 else 3))

def preprocess_video_worker(args):
    """Worker function for video preprocessing"""
    video_path, window_size, hop_size = args
    
    try:
        # Initialize face detection
        face_net = get_face_net()
        if face_net is None:
            logger.error(f"Face detection model not loaded for {os.path.basename(video_path)}")
            return (None, 0, os.path.basename(video_path), 0, 0)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return (None, 0, os.path.basename(video_path), 0, 0)
        
        frames = []
        all_boxes = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if not fps or fps < 1:
            fps = 30
        
        # Read frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (640, 360))
            
            # Detect faces
            boxes = detect_faces_dnn(frame, face_net)
            
            frames.append(frame)
            all_boxes.append(boxes)
            frame_count += 1
        
        cap.release()
        
        if len(frames) < window_size:
            logger.warning(f"Video {os.path.basename(video_path)} too short: {len(frames)} frames < {window_size}")
            return (None, 0, os.path.basename(video_path), 0, len(frames))
        
        # Track faces
        tracks = track_faces(all_boxes)
        
        if not tracks:
            logger.warning(f"No valid face tracks found in {os.path.basename(video_path)}")
            return (None, 0, os.path.basename(video_path), 0, len(frames))
        
        # Process each track
        rppg_features_all_tracks = []
        
        for track_id, track in tracks.items():
            try:
                # Create per-frame boxes array
                per_frame_boxes = [None] * len(frames)
                for (frame_idx, box) in track:
                    if frame_idx < len(frames):
                        per_frame_boxes[frame_idx] = box
                
                # Extract RGB signal
                rgb_signal, face_mask = extract_rgb_signal_track(frames, per_frame_boxes, AUGMENTATION)
                
                # Create sliding windows
                rgb_windows = sliding_windows_with_mask(rgb_signal, face_mask, window_size, hop_size, min_face_ratio=0.8)
                
                # Process each window
                for win in rgb_windows:
                    if win.size > 0:
                        rppg_feat, mean_bpm, rppg_sig_chrom, f, pxx = compute_rppg_features_multi(win, fps)
                        if rppg_feat is not None and rppg_feat.size > 0:
                            rppg_features_all_tracks.append(rppg_feat)
            
            except Exception as e:
                logger.warning(f"Track processing failed for track {track_id} in {os.path.basename(video_path)}: {e}")
                continue
        
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
            logger.warning(f"No valid features extracted from {os.path.basename(video_path)}")
            return (None, 0, os.path.basename(video_path), 0, len(frames))
    
    except Exception as e:
        logger.error(f"Video processing failed for {video_path}: {e}")
        return (None, 0, os.path.basename(video_path), 0, 0)

def cache_batches_parallel(video_dir, label, class_idx, batch_size=128, cache_dir='cache/batches/train',
                          window_size=150, hop_size=75, max_workers=None, start_batch_idx=0):
    """Process videos in parallel and cache batches"""
    try:
        # Validate inputs
        if not os.path.exists(video_dir):
            logger.error(f"Video directory does not exist: {video_dir}")
            return
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get video files
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        file_list = []
        
        for f in os.listdir(video_dir):
            if f.lower().endswith(video_extensions):
                video_path = os.path.join(video_dir, f)
                if os.path.isfile(video_path):
                    file_list.append(video_path)
        
        file_list = sorted(file_list)
        
        if not file_list:
            logger.error(f"No video files found in {video_dir}")
            return
        
        logger.info(f"Found {len(file_list)} video files for {label}")
        
        # Initialize batch storage
        X_feat, y = [], []
        batch_idx = start_batch_idx
        
        # Create worker tasks
        tasks = [(video_path, window_size, hop_size) for video_path in file_list]
        
        # Determine number of workers
        if max_workers:
            n_workers = max_workers
        else:
            n_workers = max(1, min(multiprocessing.cpu_count() - 1, len(file_list)))
        
        logger.info(f"Using {n_workers} workers for {label} ({len(file_list)} videos)")
        
        # Process videos in parallel
        processed_count = 0
        failed_count = 0
        
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
                    logger.warning(f"{vname}: No valid features extracted (frames: {n_frames})")
                    failed_count += 1
                    continue
                
                processed_count += 1
                X_feat.append(feats)
                y.append(np.full((feats.shape[0],), class_idx, dtype=np.int32))
                
                # Check if we have enough samples for a batch
                total_samples = sum([arr.shape[0] for arr in X_feat])
                
                while total_samples >= batch_size:
                    # Collect samples for current batch
                    Xf, Y = [], []
                    collected = 0
                    
                    while X_feat and collected < batch_size:
                        needed = batch_size - collected
                        take = min(needed, X_feat[0].shape[0])
                        
                        Xf.append(X_feat[0][:take])
                        Y.append(y[0][:take])
                        
                        if take < X_feat[0].shape[0]:
                            # Partial consumption
                            X_feat[0] = X_feat[0][take:]
                            y[0] = y[0][take:]
                        else:
                            # Full consumption
                            X_feat.pop(0)
                            y.pop(0)
                        
                        collected += take
                    
                    # Stack and save batch
                    Xf = np.concatenate(Xf, axis=0)
                    Y = np.concatenate(Y, axis=0)
                    
                    batch_path_x = os.path.join(cache_dir, f"{label}_Xrppg_batch_{batch_idx}.npy")
                    batch_path_y = os.path.join(cache_dir, f"{label}_y_batch_{batch_idx}.npy")
                    
                    np.save(batch_path_x, Xf)
                    np.save(batch_path_y, Y)
                    
                    logger.info(f"Saved batch {batch_idx}: {Xf.shape[0]} samples")
                    batch_idx += 1
                    total_samples -= batch_size
        
        # Save remaining samples as final batch
        if X_feat:
            Xf = np.concatenate(X_feat, axis=0)
            Y = np.concatenate(y, axis=0)
            
            if len(Xf) > 0:
                batch_path_x = os.path.join(cache_dir, f"{label}_Xrppg_batch_{batch_idx}.npy")
                batch_path_y = os.path.join(cache_dir, f"{label}_y_batch_{batch_idx}.npy")
                
                np.save(batch_path_x, Xf)
                np.save(batch_path_y, Y)
                
                logger.info(f"Saved final batch {batch_idx}: {Xf.shape[0]} samples")
        
        logger.info(f"Completed {label}: {processed_count} videos processed, {failed_count} failed")
        
    except Exception as e:
        logger.error(f"Batch caching failed: {e}")
        raise

def validate_paths():
    """Validate that required model files exist"""
    face_proto = 'models/weights-prototxt.txt'
    face_model = 'models/res_ssd_300Dim.caffeModel'
    
    if not os.path.exists(face_proto):
        logger.error(f"Face detection prototxt not found: {face_proto}")
        return False
    
    if not os.path.exists(face_model):
        logger.error(f"Face detection model not found: {face_model}")
        return False
    
    return True

if __name__ == "__main__":
    # Validate required files
    if not validate_paths():
        logger.error("Please ensure face detection model files are available in 'models/' directory")
        exit(1)
    
    # Example usage - update paths according to your setup
    try:
        cache_batches_parallel(
            video_dir='path/to/your/real/videos',
            label='real',
            class_idx=0,
            batch_size=64,
            cache_dir='cache/batches/train/real',
            window_size=150,
            hop_size=75,
            start_batch_idx=0
        )
        
        # Uncomment to process fake videos
        # cache_batches_parallel(
        #     video_dir='path/to/your/fake/videos',
        #     label='fake',
        #     class_idx=1,
        #     batch_size=64,
        #     cache_dir='cache/batches/train/fake',
        #     window_size=150,
        #     hop_size=75,
        #     start_batch_idx=0
        # )
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise