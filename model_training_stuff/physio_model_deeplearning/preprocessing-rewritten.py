import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import detrend, butter, filtfilt, periodogram
import logging
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import re
import threading
import uuid
import scipy.stats
import random
import mediapipe as mp
from sklearn.decomposition import PCA
import argparse
import gc

# ==================== CONFIGURATION =========================

# Model paths (edit as needed)
FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'

# 5 ROIs with MediaPipe 468 landmarks indices (adjustable)
ROI_INDICES = {
    'left_cheek': [207, 216, 206, 203, 129, 209, 126, 47, 121,
                   120, 119, 118, 117, 111, 116, 123, 147, 187],
    'right_cheek': [350, 277, 355, 429, 279, 331, 423, 426, 436, 427,
                    411, 376, 352, 345, 340, 346, 347, 348, 349],
    'forehead':    [10, 338, 297, 332, 284, 251, 301, 300, 293, 
                    334, 296, 336, 9, 107, 66, 105, 63, 70, 71, 21, 
                    54, 103, 67, 109],
    'chin':        [43, 202, 210, 169, 150, 149, 176, 148, 152, 377, 400, 378, 379,
                    394, 430, 422, 273, 335, 406, 313, 18, 83, 182, 106],
    'nose':        [168, 122, 174, 198, 49, 48, 115, 220, 44, 1, 274, 440,
                    344, 279, 429, 399, 351]
}

WINDOW_SIZE = 150  # Frames per window (e.g. 5s @ 30fps)
HOP_SIZE = 75      # Overlap step (e.g. 50% overlap)
BATCH_SIZE = 40    # Videos per output HDF5 file
MIN_REAL_FRAMES = int(0.6 * WINDOW_SIZE)  # Reject if <60% valid frames

# Suppress MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("physio_preproc")

# ================== FACE DETECTION & TRACKING ===============

def load_face_net(proto, model):
    try:
        net = cv2.dnn.readNetFromCaffe(proto, model)
        return net
    except Exception as e:
        logger.warning(f"Loading Caffe Model Failed: {e}")
        return None

def detect_faces(frame, net, conf=0.5):
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

# =============== ROI & rPPG EXTRACTION =======================

# Thread/process-local storage for FaceMesh
_thread_local = threading.local()

def get_facemesh():
    """Get process-local FaceMesh instance for multiprocessing safety"""
    if not hasattr(_thread_local, "mp_face_mesh"):
        _thread_local.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
    return _thread_local.mp_face_mesh

def cleanup_facemesh():
    """Optional cleanup function for MediaPipe resources"""
    if hasattr(_thread_local, "mp_face_mesh"):
        _thread_local.mp_face_mesh.close()
        del _thread_local.mp_face_mesh

def extract_landmarks(frame, box):
    """
    Extract 468 facial landmarks (pixel coordinates) for the face inside 'box' in 'frame' using MediaPipe FaceMesh.
    Returns: list of (x, y) tuples (length 468), or None if detection fails.
    """
    try:
        face_mesh = get_facemesh()
        x, y, w, h = box
        
        # Crop ROI for the face
        roi = frame[y:y+h, x:x+w]
        if roi.shape[0] < 40 or roi.shape[1] < 40:  # Avoid too-small faces
            return None
        
        # Convert to RGB (MediaPipe expects RGB)
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_roi)
        
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            
            # Map landmark coords from ROI back to original image coordinates
            pts = []
            for lm in lms:
                lx = int(lm.x * w) + x
                ly = int(lm.y * h) + y
                pts.append((lx, ly))
            
            if len(pts) == 468:
                return pts
        
        return None  # If no landmarks or wrong count
    
    except Exception as e:
        logger.warning(f"MediaPipe landmark extraction failed: {e}")
        return None

def extract_roi_means(frame, box, landmarks, rois=ROI_INDICES):
    """
    Extract mean RGB values for each ROI using MediaPipe landmarks.
    Fixed to handle None landmarks and proper coordinate mapping.
    """
    rgb_means = {}
    
    # Handle case where landmarks failed
    if landmarks is None:
        return {roi: [0.0, 0.0, 0.0] for roi in rois}
    
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]
    
    for roi, idxs in rois.items():
        try:
            # Get landmark points for this ROI
            pts = []
            for i in idxs:
                if i < len(landmarks):
                    lx, ly = landmarks[i]
                    # Ensure coordinates are within frame bounds
                    lx = max(0, min(lx, frame_w - 1))
                    ly = max(0, min(ly, frame_h - 1))
                    pts.append([lx, ly])
            
            pts = np.array(pts, np.int32)
            
            if len(pts) > 2:
                # Create mask for this ROI
                mask = np.zeros((frame_h, frame_w), np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                
                # Extract RGB means from masked region
                vals = []
                for ch in range(3):
                    channel_vals = frame[:, :, ch][mask == 1]
                    if len(channel_vals) > 0:
                        vals.append(float(np.mean(channel_vals)))
                    else:
                        vals.append(0.0)
                
                rgb_means[roi] = vals
            else:
                rgb_means[roi] = [0.0, 0.0, 0.0]
                
        except Exception as e:
            logger.warning(f"ROI extraction failed for {roi}: {e}")
            rgb_means[roi] = [0.0, 0.0, 0.0]
    
    return rgb_means

def rppg_chrom(rgb):
    """CHROM algorithm for rPPG signal extraction"""
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] < 10:
        return np.zeros(S.shape[0])
    
    # Normalize
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    
    # CHROM transformation
    h = S[:, 1] - S[:, 0]  # Green - Red
    s = S[:, 1] + S[:, 0] - 2 * S[:, 2]  # Green + Red - 2*Blue
    
    return (h + s).astype(np.float32)

def rppg_pos(rgb):
    """POS algorithm for rPPG signal extraction"""
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] < 10:
        return np.zeros(S.shape[0])
    
    # Normalize by temporal mean
    S_norm = S / (S.mean(0) + 1e-6)
    
    # POS projection matrix
    P = np.array([[0, 1, -1], [-2, 1, 1]], np.float32)
    Y = S_norm @ P.T
    
    # Normalize by standard deviation
    std_y = Y.std(0) + 1e-6
    Y_norm = Y / std_y
    
    # Adaptive ratio
    alpha = Y_norm[:, 0].std() / (Y_norm[:, 1].std() + 1e-6)
    rppg = Y_norm[:, 0] - alpha * Y_norm[:, 1]
    
    return rppg.astype(np.float32)

def rppg_pca(rgb):
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] < 10:
        return rppg_chrom(rgb)  # Fallback to CHROM
    
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
    Apply light preprocessing to rPPG signals with improved stability
    """
    signal = np.array(signal, dtype=np.float32)
    
    if signal.size == 0 or len(signal) < 10:
        return signal
    
    try:
        # 1. Detrend (remove linear drift)
        processed = detrend(signal, type='linear').astype(np.float32)
        
        # 2. Band-pass filter with comprehensive stability checks
        if apply_filtering and len(signal) > 30:
            filter_order = 3
            
            # Critical: Check minimum signal length for filtfilt
            # filtfilt needs signal length >= 3 * filter_order for stability
            min_length_for_filtfilt = 3 * filter_order * 2  # Conservative: 18 samples
            
            if len(signal) >= min_length_for_filtfilt:
                nyquist = fs / 2.0
                low_cutoff = 0.5 / nyquist   # 30 BPM
                high_cutoff = 5.0 / nyquist  # 300 BPM
                
                # Ensure cutoffs are valid for filter design
                low_cutoff = max(0.01, min(low_cutoff, 0.45))
                high_cutoff = max(0.55, min(high_cutoff, 0.99))
                
                # Additional validation: ensure high > low and both in valid range
                if low_cutoff < high_cutoff and high_cutoff < 1.0 and low_cutoff > 0:
                    try:
                        # Design filter with stability check
                        b, a = butter(filter_order, [low_cutoff, high_cutoff], btype='band')
                        
                        # Check filter stability (poles inside unit circle)
                        poles = np.roots(a)
                        if np.all(np.abs(poles) < 1.0):
                            # Apply filter with edge padding for stability
                            padlen = min(len(processed) // 3, filter_order * 3)
                            processed = filtfilt(b, a, processed, padlen=padlen).astype(np.float32)
                        else:
                            logger.warning("Unstable filter detected, skipping filtering")
                            
                    except Exception as e:
                        logger.warning(f"Filtering failed: {e}, using alternative method")
                        # Fallback: simple moving average filter
                        window_size = max(3, int(fs / 5))  # ~0.2 second window
                        if len(processed) >= window_size:
                            processed = np.convolve(processed, np.ones(window_size)/window_size, mode='same').astype(np.float32)
            else:
                logger.info(f"Signal too short for filtfilt ({len(signal)} < {min_length_for_filtfilt}), skipping filtering")
        
        # 3. Robust normalization with outlier protection
        # Remove extreme outliers first (beyond 5 standard deviations)
        mean_val = np.mean(processed)
        std_val = np.std(processed)
        
        if std_val > 1e-6:
            # Clip extreme outliers
            outlier_threshold = 5 * std_val
            processed = np.clip(processed, mean_val - outlier_threshold, mean_val + outlier_threshold)
            
            # Robust normalization using median and MAD
            median_val = np.median(processed)
            mad = np.median(np.abs(processed - median_val))
            
            if mad > 1e-6:
                processed = ((processed - median_val) / (1.4826 * mad)).astype(np.float32)
            else:
                # Fallback to standard normalization
                processed = ((processed - mean_val) / std_val).astype(np.float32)
        else:
            logger.warning("Signal has zero variance, returning zeros")
            processed = np.zeros_like(processed, dtype=np.float32)
        
        return processed
        
    except Exception as e:
        logger.warning(f"Signal preprocessing failed: {e}, returning original")
        return signal.astype(np.float32)

def pad_mask(arr, win):
    """Pad array to window size and create mask"""
    L = len(arr)
    pad = win - L
    mask = np.concatenate([np.ones(L), np.zeros(max(pad, 0))])
    arr = np.concatenate([arr, np.zeros(max(pad, 0))])
    return arr[:win].astype(np.float32), mask[:win].astype(np.float32)

def augment_signal(sig, aug=None, strength=0.1, mask=None):
    """Apply data augmentation to signals"""
    if aug is None or sig.size == 0:
        return sig.astype(np.float32)
    
    real_idx = mask > 0.5 if mask is not None else np.ones(sig.shape[0], bool)
    
    if aug == 'noise':
        sig[real_idx] += np.random.randn(real_idx.sum()) * strength * sig[real_idx].std()
    elif aug == 'amplitude':
        sig[real_idx] *= (1 + strength * np.sin(np.linspace(0, 2*np.pi, real_idx.sum())))
    elif aug == 'frequency':
        x = np.arange(real_idx.sum())
        shift = 1 + strength * np.random.uniform(-1, 1)
        sig[real_idx] = np.interp(x, x * shift, sig[real_idx], left=0, right=0)
    elif aug == 'phase':
        shift = int(strength * real_idx.sum() * np.random.rand())
        sig[real_idx] = np.roll(sig[real_idx], shift)
    elif aug == 'dropout' and real_idx.sum() > 5:
        l = max(1, int(strength * 0.25 * real_idx.sum()))
        i = np.random.randint(0, real_idx.sum() - l + 1)
        sig[real_idx][i:i+l] = 0
    elif aug == 'outlier' and real_idx.sum() > 5:
        idx = np.random.choice(real_idx.sum(), size=1)
        sig[real_idx][idx] += 8 * sig[real_idx].std()
    
    return sig.astype(np.float32)

def extract_window_signals(frames, boxes, start, end, fps=30, augment=True):
    """
    Extract rPPG signals from video window using MediaPipe landmarks.
    Fixed to properly handle failed landmark detection.
    """
    window_frames = frames[start:end]
    window_boxes = boxes[start:end]
    window_mask = [1 if b is not None else 0 for b in window_boxes]
    
    # Extract ROI signals for each frame
    roi_sigs = {roi: [] for roi in ROI_INDICES}
    
    for f, b in zip(window_frames, window_boxes):
        if b is not None:
            # Extract landmarks using MediaPipe
            lms = extract_landmarks(f, b)
            
            # Extract ROI means (handles None landmarks gracefully)
            roi_rgb = extract_roi_means(f, b, lms)
            
            for roi in ROI_INDICES:
                roi_sigs[roi].append(roi_rgb[roi])
        else:
            # No face detected - append zeros
            for roi in ROI_INDICES:
                roi_sigs[roi].append([0.0, 0.0, 0.0])
    
    # Apply rPPG algorithms to each ROI
    multi_roi = {}
    for roi, sig in roi_sigs.items():
        # Extract rPPG signals using different methods
        chrom = rppg_chrom(sig)
        pos = rppg_pos(sig)
        ica = rppg_pca(sig)

        chrom = apply_signal_preprocessing(chrom, fps)
        pos = apply_signal_preprocessing(pos, fps)
        ica = apply_signal_preprocessing(ica, fps)

        # Apply augmentation if requested
        if augment:
            if random.random() < 0.6: # CHANGE TO BE LOWER FOR FAKE, HIGHER FOR REAL
                augs = ['noise', 'amplitude', 'frequency', 'phase']
                aug = random.choice(augs)
                chrom = augment_signal(chrom, aug, strength=0.12, mask=np.array(window_mask))
                pos = augment_signal(pos, aug, strength=0.12, mask=np.array(window_mask))
                ica = augment_signal(ica, aug, strength=0.12, mask=np.array(window_mask))
            elif random.random() < 0.15:
                for s in [chrom, pos, ica]:
                    s = augment_signal(s, 'dropout', strength=0.1, mask=np.array(window_mask))
        
        # Pad to window size
        chrom, _ = pad_mask(chrom, WINDOW_SIZE)
        pos, _ = pad_mask(pos, WINDOW_SIZE)
        ica, _ = pad_mask(ica, WINDOW_SIZE)
        
        multi_roi[roi] = {'chrom': chrom, 'pos': pos, 'ica': ica}
    
    return multi_roi, np.array(window_mask, np.float32)

def interpolate_corrupted_frames(frames):
    """
    Replace None or corrupted frames with interpolation of prev/next valid frames.
    Args:
        frames: list of frames (some are None)
    Returns:
        new_frames: list of same length, with all frames valid
    """
    n = len(frames)
    valid_indices = [i for i, f in enumerate(frames) if f is not None]

    if not valid_indices:
        raise ValueError("No valid frames found in video.")

    # Make a copy to avoid modifying in-place
    out_frames = list(frames)

    for idx in range(n):
        if out_frames[idx] is not None:
            continue  # Already valid

        # Find previous and next valid frames
        prev_idx = idx - 1
        while prev_idx >= 0 and out_frames[prev_idx] is None:
            prev_idx -= 1

        next_idx = idx + 1
        while next_idx < n and out_frames[next_idx] is None:
            next_idx += 1

        prev_frame = out_frames[prev_idx] if prev_idx >= 0 else None
        next_frame = out_frames[next_idx] if next_idx < n else None

        # Interpolate or fallback
        if prev_frame is not None and next_frame is not None:
            # Linear average interpolation
            interp = cv2.addWeighted(prev_frame, 0.5, next_frame, 0.5, 0)
            out_frames[idx] = interp
        elif prev_frame is not None:
            out_frames[idx] = prev_frame.copy()
        elif next_frame is not None:
            out_frames[idx] = next_frame.copy()
        else:
            # No valid frames anywhere (should never happen)
            h, w, c = frames[valid_indices[0]].shape
            out_frames[idx] = np.zeros((h, w, c), dtype=frames[valid_indices[0]].dtype)

    return out_frames

# ============== ML FEATURE ENGINEERING (STATS) ==============
ROIS = ['left_cheek', 'right_cheek', 'forehead', 'chin', 'nose']
METHODS = ['chrom', 'pos', 'ica']
STATS = [
    'mean', 'std', 'var', 'min', 'max', 'median', 'range',
    'q25', 'q75', 'iqr', 'skew', 'kurt',
    'bandpower', 'spectral_centroid', 'zero_crossing_rate', 'energy', 'rms', 'snr_estimate'
]

EXPECTED_FEATURE_KEYS = [
    f"{roi}_{method}_{stat}"
    for roi in ROIS
    for method in METHODS
    for stat in STATS
] + ['valid_frames']

def pad_ml_features(feats, expected_keys=EXPECTED_FEATURE_KEYS):
    return {k: float(feats.get(k, 0.0)) for k in expected_keys}

def extract_ml_feats(multi_roi, window_mask, fs=30):
    """Extract comprehensive ML features from rPPG signals"""
    
    def _bandpower(sig, fs, band=(0.7, 4.0)):
        """Compute bandpower in the heart rate frequency range"""
        if len(sig) < 10:
            return 0.0
        try:
            f, Pxx = periodogram(sig, fs=fs)
            idx = (f >= band[0]) & (f <= band[1])
            return float(np.sum(Pxx[idx])) if np.any(idx) else 0.0
        except:
            return 0.0
    
    def _spectral_centroid(sig, fs):
        """Compute spectral centroid"""
        if len(sig) < 10:
            return 0.0
        try:
            f, Pxx = periodogram(sig, fs=fs)
            total_power = np.sum(Pxx)
            if total_power < 1e-10:
                return 0.0
            return float(np.sum(f * Pxx) / total_power)
        except:
            return 0.0
    
    def _zero_crossing_rate(sig):
        """Compute zero crossing rate"""
        if len(sig) < 2:
            return 0.0
        try:
            centered = sig - np.mean(sig)
            crossings = np.sum(np.diff(np.sign(centered)) != 0)
            return float(crossings / len(sig))
        except:
            return 0.0
    
    feats = {}
    
    for roi in multi_roi:
        for method in ['chrom', 'pos', 'ica']:
            # Get valid signal points
            sig = multi_roi[roi][method][window_mask > 0.5]
            if len(sig) < 10:
                continue
            
            # Basic statistics
            feats[f'{roi}_{method}_mean'] = float(np.mean(sig))
            feats[f'{roi}_{method}_std'] = float(np.std(sig))
            feats[f'{roi}_{method}_var'] = float(np.var(sig))
            feats[f'{roi}_{method}_min'] = float(np.min(sig))
            feats[f'{roi}_{method}_max'] = float(np.max(sig))
            feats[f'{roi}_{method}_median'] = float(np.median(sig))
            feats[f'{roi}_{method}_range'] = float(np.max(sig) - np.min(sig))
            
            # Percentiles
            feats[f'{roi}_{method}_q25'] = float(np.percentile(sig, 25))
            feats[f'{roi}_{method}_q75'] = float(np.percentile(sig, 75))
            feats[f'{roi}_{method}_iqr'] = float(np.percentile(sig, 75) - np.percentile(sig, 25))
            
            # Higher order moments
            feats[f'{roi}_{method}_skew'] = float(scipy.stats.skew(sig))
            feats[f'{roi}_{method}_kurt'] = float(scipy.stats.kurtosis(sig))
            
            # Frequency domain features
            feats[f'{roi}_{method}_bandpower'] = _bandpower(sig, fs)
            feats[f'{roi}_{method}_spectral_centroid'] = _spectral_centroid(sig, fs)
            
            # Time domain features
            feats[f'{roi}_{method}_zero_crossing_rate'] = _zero_crossing_rate(sig)
            feats[f'{roi}_{method}_energy'] = float(np.sum(sig**2))
            feats[f'{roi}_{method}_rms'] = float(np.sqrt(np.mean(sig**2)))
            
            # Signal quality indicators
            feats[f'{roi}_{method}_snr_estimate'] = float(np.var(sig) / max(np.var(np.diff(sig)), 1e-10))
    
    feats['valid_frames'] = float(window_mask.sum())
    return feats

# ============= VIDEO WINDOW EXTRACTION & QC ==================

def segment_windows(nframes, win=WINDOW_SIZE, hop=HOP_SIZE):
    """Generate sliding windows over video frames"""
    return [(s, min(s + win, nframes)) for s in range(0, max(nframes - win + 1, 1), hop)]

def quick_qc(frames, boxes, start, end):
    """Quick quality check: require at least 2 valid boxes"""
    return sum(1 for b in boxes[start:end] if b is not None) >= 2

def process_video(video_path, augment=True):
    """
    Process a single video file and extract rPPG windows.
    Now uses real MediaPipe landmark detection.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            logger.warning(f"Invalid FPS detected: {fps}, defaulting to 30")
            fps = 30.0

        frames = []
        face_net = load_face_net(FACE_PROTO, FACE_MODEL)

        # Read ALL frames (including None for corrupted frames)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Always append, even if frame is None
            frames.append(frame)
        cap.release()

        # Interpolate/fill corrupted frames
        frames = interpolate_corrupted_frames(frames)

        # Detect faces on all frames (now all valid)
        boxes_seq = [detect_faces(frame, face_net) for frame in frames]

        if len(frames) < WINDOW_SIZE:
            return None
        
        # Track faces across frames
        tracks = robust_track_faces(boxes_seq)
        
        windows_all = []
        for tid, track in tracks.items():
            # Create per-frame box mapping
            per_frame_boxes = [None] * len(frames)
            for fidx, box in track:
                per_frame_boxes[fidx] = box
            
            # Extract windows for this face track
            for start, end in segment_windows(len(frames)):
                if not quick_qc(frames, per_frame_boxes, start, end):
                    continue
                
                # Extract rPPG signals using MediaPipe
                multi_roi, window_mask = extract_window_signals(
                    frames, per_frame_boxes, start, end, fps=fps, augment=augment
                )
                
                if window_mask.sum() < MIN_REAL_FRAMES:
                    continue
                
                # Extract ML features
                feats = extract_ml_feats(multi_roi, window_mask, fs=fps)
                feats = pad_ml_features(feats)
                
                windows_all.append({
                    'multi_roi': multi_roi,
                    'window_mask': window_mask,
                    'ml_features': feats
                })
        
        return windows_all
    
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return None
    
    finally:
        # Clean up MediaPipe resources
        cleanup_facemesh()
        try:
            del frames, face_net, boxes_seq
        except Exception:
            pass
        gc.collect()

# ============== BATCH STORAGE (HDF5) =======================

def safe_group_name(filename):
    """Generate safe HDF5 group name from filename"""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r'[^\w\-_.]', '_', name)
    if name and name[0].isdigit():
        name = 'v_' + name
    return name[:128] + '_' + str(uuid.uuid4())[:8]

def save_batch_to_hdf5(batch_data, out_path, label):
    """Save batch of processed videos to HDF5 file"""
    with h5py.File(out_path, 'w') as f:
        f.attrs['dataset_label'] = label
        
        for vid, windows in batch_data.items():
            if not windows:
                continue
                
            gname = safe_group_name(vid)
            g = f.create_group(gname)
            g.attrs['original_filename'] = vid
            
            for i, win in enumerate(windows):
                wg = g.create_group(f'window_{i}')
                
                # Save multi_roi signals
                roi_g = wg.create_group('multi_roi')
                for roi, signals in win['multi_roi'].items():
                    roi_sub = roi_g.create_group(roi)
                    for method, arr in signals.items():
                        roi_sub.create_dataset(method, data=arr, compression='gzip', compression_opts=4)
                
                # Save masks
                wg.create_dataset('window_mask', data=win['window_mask'], compression='gzip')
                
                # Save ML features
                feat_g = wg.create_group('ml_features')
                for k, v in win['ml_features'].items():
                    feat_g.create_dataset(k, data=v)

# =================== MAIN MULTIPROC/PROCESSING ===================

def get_video_files(video_dir, exts=('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
    """Get all video files from directory"""
    return [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
            if f.lower().endswith(exts)]

def get_next_batch_idx(output_dir, label):
    # Find all batch files matching pattern
    batch_files = [f for f in os.listdir(output_dir) if f.startswith(f"{label}_batch") and f.endswith("_raw.h5")]
    max_idx = -1
    for fname in batch_files:
        m = re.match(rf"{label}_batch(\d+)_raw\.h5", fname)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1  # next unused batch number

def main():
    parser = argparse.ArgumentParser(description='Physio rPPG Deepfake Preprocessing')
    # ----- CHANGE INPUT OUTPUT IN SCRIPT ACCORDINGLY -----
    parser.add_argument('--input', type=str, help='Input video folder', default="G:/deepfake_training_datasets/Physio_Model/VALIDATION/real")
    parser.add_argument('--output', type=str, help='Output batch folder', default="D:/model_training/cache/batches/physio-deep-v1/real")
    # ----- CHANGE LABEL IN CLI USING --label fake OR --label real
    parser.add_argument('--label', type=str, required=True, help='Label for this set (real/fake)')
    # ----- DO NOT NEED TO CHANGE
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max_workers', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    videos = get_video_files(args.input)
    logger.info(f"Found {len(videos)} videos in {args.input}")

    # Skip already-processed videos
    processed = set()
    log_file = os.path.join(args.output, f'{args.label}_processed.txt')
    failure_log = os.path.join(args.output, f'{args.label}_failed.txt')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            processed = set(line.strip() for line in f)

    # Skip videos listed in failed.txt
    failed = set()
    if os.path.exists(failure_log):
        with open(failure_log, 'r') as f:
            failed = set(line.strip() for line in f)

    videos = [
        v for v in videos
        if os.path.basename(v) not in processed
        and os.path.basename(v) not in failed
    ]


    logger.info(f"Processing {len(videos)} new videos (label: {args.label})")

    # --- Find the next batch index ---
    batch_idx = get_next_batch_idx(args.output, args.label)
    n_in_batch = 0
    batch_data = {}

    with ProcessPoolExecutor(max_workers=args.max_workers) as exe:
        future_to_vid = {exe.submit(process_video, v): v for v in videos}
        
        for fut in tqdm(as_completed(future_to_vid), total=len(future_to_vid), desc='Videos'):
            vid = future_to_vid[fut]
            try:
                windows = fut.result()
            except Exception as e:
                logger.warning(f"Error processing {vid}: {e}")
                windows = None
            
            if windows and len(windows) > 0:
                batch_data[vid] = windows
                n_in_batch += 1
                logger.info(f"{os.path.basename(vid)}: {len(windows)} windows")

                # Mark as processed
                with open(log_file, 'a') as f:
                    f.write(os.path.basename(vid) + '\n')
            else:
                logger.warning(f"No valid windows from {vid}")
                # Log failures for pruning
                with open(failure_log, 'a') as f:
                    f.write(os.path.basename(vid) + '\n')

            # Save batch when it reaches the specified size
            if n_in_batch >= args.batch_size:
                out_path = os.path.join(args.output, f'{args.label}_batch{batch_idx}_raw.h5')
                save_batch_to_hdf5(batch_data, out_path, args.label)
                logger.info(f"Saved batch {batch_idx} ({n_in_batch} videos) to {out_path}")
                batch_data = {}
                n_in_batch = 0
                batch_idx += 1
        
        # Save final batch if there's remaining data
        if batch_data:
            out_path = os.path.join(args.output, f'{args.label}_batch{batch_idx}_raw.h5')
            save_batch_to_hdf5(batch_data, out_path, args.label)
            logger.info(f"Saved final batch {batch_idx} ({n_in_batch} videos) to {out_path}")

if __name__ == '__main__':
    main()