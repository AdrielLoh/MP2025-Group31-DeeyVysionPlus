import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy.signal import detrend, butter, filtfilt, periodogram
import scipy.stats
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

# === rPPG multi-method and extended feature extraction ===
def safe_normalize(arr, axis=None, epsilon=1e-8):
    """Safely normalize array"""
    std = np.std(arr, axis=axis)
    return arr / (std + epsilon)

def rppg_chrom(rgb):
    """CHROM method for rPPG"""
    try:
        if rgb.size == 0 or rgb.shape[0] < 2:
            return np.zeros(rgb.shape[0])
        
        S = rgb.copy()
        h = safe_normalize(S[:, 1] - S[:, 0])
        s = safe_normalize(S[:, 1] + S[:, 0] - 2 * S[:, 2])
        return h + s
    except Exception as e:
        logger.warning(f"CHROM method failed: {e}")
        return np.zeros(rgb.shape[0])

def rppg_pos(rgb):
    """POS method for rPPG"""
    try:
        if rgb.size == 0 or rgb.shape[0] < 2:
            return np.zeros(rgb.shape[0])
        
        S = rgb.copy()
        S_mean = np.mean(S, axis=0)
        S_mean[S_mean == 0] = 1e-8
        S = S / S_mean
        
        Xcomp = 3 * S[:, 0] - 2 * S[:, 1]
        Ycomp = 1.5 * S[:, 0] + S[:, 1] - 1.5 * S[:, 2]
        Ycomp[Ycomp == 0] = 1e-8
        
        return Xcomp / Ycomp
    except Exception as e:
        logger.warning(f"POS method failed: {e}")
        return np.zeros(rgb.shape[0])

def rppg_green(rgb):
    """Green channel method for rPPG"""
    try:
        if rgb.size == 0:
            return np.zeros(rgb.shape[0])
        return rgb[:, 1]
    except Exception as e:
        logger.warning(f"Green method failed: {e}")
        return np.zeros(rgb.shape[0])

def compute_band_powers(f, pxx, bands):
    """Compute power in frequency bands"""
    try:
        if len(f) == 0 or len(pxx) == 0:
            return [0] * (len(bands) * 2)
        
        total_power = np.sum(pxx)
        if total_power == 0:
            return [0] * (len(bands) * 2)
        
        power_features = []
        for low, high in bands:
            mask = (f >= low) & (f <= high)
            power = np.sum(pxx[mask])
            ratio = power / total_power
            power_features.extend([power, ratio])
        return power_features
    except Exception as e:
        logger.warning(f"Band power computation failed: {e}")
        return [0] * (len(bands) * 2)

def compute_autocorrelation(sig):
    """Compute autocorrelation peak"""
    try:
        if len(sig) < 2:
            return 0
        
        sig_centered = sig - np.mean(sig)
        if np.std(sig_centered) == 0:
            return 0
        
        acf = np.correlate(sig_centered, sig_centered, mode='full')
        acf = acf[acf.size // 2:]
        
        if len(acf) <= 1 or acf[0] == 0:
            return 0
        
        peak_lag = np.argmax(acf[1:]) + 1
        return acf[peak_lag] / acf[0]
    except Exception as e:
        logger.warning(f"Autocorrelation computation failed: {e}")
        return 0

def compute_entropy(sig):
    """Compute signal entropy"""
    try:
        if len(sig) < 2:
            return 0
        
        hist, _ = np.histogram(sig, bins=32, density=True)
        hist = hist + 1e-8
        return -np.sum(hist * np.log2(hist))
    except Exception as e:
        logger.warning(f"Entropy computation failed: {e}")
        return 0

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
        return filtfilt(b, a, signal)
    except Exception as e:
        logger.warning(f"Bandpass filter failed: {e}")
        return signal

def compute_rppg_features_multi(rgb_signal, fs):
    """Compute rPPG features using multiple methods"""
    try:
        if rgb_signal.size == 0 or rgb_signal.shape[0] < 10:
            return np.zeros(57, dtype=np.float32), 0, np.zeros(rgb_signal.shape[0]), np.array([]), np.array([])
        
        # Preprocess signal
        rgb_detrend = detrend(rgb_signal, axis=0)
        rgb_norm = rgb_detrend.copy()
        
        # Normalize each channel
        for i in range(rgb_norm.shape[1]):
            std = np.std(rgb_norm[:, i])
            if std > 0:
                rgb_norm[:, i] = (rgb_norm[:, i] - np.mean(rgb_norm[:, i])) / std
        
        # Apply rPPG methods
        rppg_methods = {
            "CHROM": rppg_chrom(rgb_norm),
            "POS": rppg_pos(rgb_norm),
            "GREEN": rppg_green(rgb_norm),
        }
        
        # Frequency bands for analysis
        bands = [(0.7, 2.5), (0.2, 0.6), (4.0, 8.0)]
        
        all_features = []
        bpm_list = []
        rppg_sig_chrom = np.zeros(len(rgb_signal))
        f, pxx = np.array([]), np.array([])
        
        for key, rppg_sig in rppg_methods.items():
            if key == "CHROM":
                rppg_sig_chrom = rppg_sig.copy()
            
            if len(rppg_sig) <= 21:
                all_features.extend([0] * 19)
                bpm_list.append(0)
                continue
            
            # Filter signal
            sig_filt = butter_bandpass_filter(rppg_sig, fs)
            
            # Compute periodogram
            try:
                f_temp, pxx_temp = periodogram(sig_filt, fs)
                valid = (f_temp >= 0.7) & (f_temp <= 4.0)
                f_temp, pxx_temp = f_temp[valid], pxx_temp[valid]
                
                if len(f_temp) == 0:
                    all_features.extend([0] * 19)
                    bpm_list.append(0)
                    continue
                
                # Find peak
                peak_idx = np.argmax(pxx_temp)
                hr_freq = f_temp[peak_idx]
                hr_bpm = hr_freq * 60
                hr_power = pxx_temp[peak_idx]
                
                # Compute SNR
                total_power = np.sum(pxx_temp)
                snr = hr_power / (total_power - hr_power + 1e-8) if total_power > hr_power else 0
                
                # Compute additional features
                band_powers = compute_band_powers(f_temp, pxx_temp, bands)
                autocorr_peak = compute_autocorrelation(sig_filt)
                entropy = compute_entropy(sig_filt)
                
                # Statistical features
                try:
                    kurt = scipy.stats.kurtosis(sig_filt, nan_policy='omit')
                    skew = scipy.stats.skew(sig_filt, nan_policy='omit')
                except:
                    kurt, skew = 0, 0
                
                # Compile features
                features = [
                    np.mean(sig_filt), np.std(sig_filt), np.max(sig_filt), np.min(sig_filt),
                    np.ptp(sig_filt), hr_freq, hr_bpm, hr_power, snr,
                    autocorr_peak, entropy, kurt, skew
                ] + band_powers
                
                all_features.extend(features)
                bpm_list.append(hr_bpm)
                
                # Store for CHROM method
                if key == "CHROM" and len(f) == 0:
                    f = f_temp
                    pxx = pxx_temp
                    
            except Exception as e:
                logger.warning(f"Feature computation failed for {key}: {e}")
                all_features.extend([0] * 19)
                bpm_list.append(0)
        
        # Compute mean BPM
        valid_bpms = [b for b in bpm_list if b > 0]
        mean_bpm = np.mean(valid_bpms) if valid_bpms else 0
        
        return (np.array(all_features, dtype=np.float32), mean_bpm, 
                rppg_sig_chrom, f, pxx)
    
    except Exception as e:
        logger.error(f"rPPG feature computation failed: {e}")
        return np.zeros(57, dtype=np.float32), 0, np.zeros(rgb_signal.shape[0]), np.array([]), np.array([])

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

def track_faces(all_face_boxes, iou_thresh=0.5):
    """Track faces across frames"""
    try:
        tracks = []
        next_track_id = 0
        
        for frame_idx, boxes in enumerate(all_face_boxes):
            if not boxes:
                continue
            
            assigned = [False] * len(boxes)
            
            # Update existing tracks
            for track in tracks:
                if frame_idx - track['last_frame'] > 5:  # Skip tracks too far behind
                    continue
                    
                best_iou = 0
                best_box_idx = -1
                
                for i, box in enumerate(boxes):
                    if not assigned[i]:
                        iou_val = iou(track['last_box'], box)
                        if iou_val > best_iou and iou_val > iou_thresh:
                            best_iou = iou_val
                            best_box_idx = i
                
                if best_box_idx >= 0:
                    track['boxes'].append((frame_idx, boxes[best_box_idx]))
                    track['last_box'] = boxes[best_box_idx]
                    track['last_frame'] = frame_idx
                    assigned[best_box_idx] = True
            
            # Create new tracks for unassigned boxes
            for i, box in enumerate(boxes):
                if not assigned[i]:
                    tracks.append({
                        'boxes': [(frame_idx, box)],
                        'last_box': box,
                        'last_frame': frame_idx,
                        'id': next_track_id
                    })
                    next_track_id += 1
        
        # Filter tracks with sufficient length
        tracklets = {t['id']: t['boxes'] for t in tracks if len(t['boxes']) > 2}
        return tracklets
    except Exception as e:
        logger.warning(f"Face tracking failed: {e}")
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
    face_proto = 'C:/Users/Adriel Loh/Documents/GitHub/MP2025-Group31-DeeyVysionPlus/models/weights-prototxt.txt'
    face_model = 'C:/Users/Adriel Loh/Documents/GitHub/MP2025-Group31-DeeyVysionPlus/models/res_ssd_300Dim.caffeModel'
    
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
            video_dir='G:/deepfake_training_datasets/Physio_Model/VALIDATION/real-aug-batch-661-onwards',
            label='real',
            class_idx=0,
            batch_size=64,
            cache_dir='D:/model_training/cache/batches/val/real',
            window_size=150,
            hop_size=75,
            start_batch_idx=175
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