import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import detrend, butter, filtfilt, periodogram
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import re
import random
from sklearn.decomposition import PCA
import argparse
import gc
import time
import tempfile

# ==================== CONFIGURATION =========================

# Model paths (edit as needed)
FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'

WINDOW_SIZE = 150  # Frames per window (e.g. 5s @ 30fps)
HOP_SIZE = 75      # Overlap step (e.g. 50% overlap)
BATCH_SIZE = 40    # Videos per output HDF5 file
MIN_REAL_FRAMES = int(0.6 * WINDOW_SIZE)  # Reject if <60% valid frames

np.random.seed(int(time.time()) % 2**32)
random.seed(int(time.time()) % 2**32)

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

# =============== rPPG EXTRACTION =======================
def extract_face_rgb_mean(frame, box):
    x, y, w, h = box
    x1, y1, x2, y2 = max(0, x), max(0, y), min(x+w, frame.shape[1]), min(y+h, frame.shape[0])
    face_region = frame[y1:y2, x1:x2]
    if face_region.size == 0:
        return [0.0, 0.0, 0.0]
    mean_rgb = np.mean(np.mean(face_region, axis=0), axis=0)
    return mean_rgb.tolist()

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
    
def rppg_green(rgb):
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[0] < 10:
        return np.zeros(S.shape[0])
    return S[:,1].astype(np.float32)  # Green channel

def rppg_pbv(rgb):
    S = np.array(rgb)
    if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] < 10:
        return np.zeros(S.shape[0])
    # Use the POS projection with extra normalization
    S_norm = S / (S.mean(0) + 1e-6)
    P = np.array([[0, 1, -1], [-2, 1, 1]], np.float32)
    Y = S_norm @ P.T
    # Project onto the orthogonal plane (orth to [1,1,1])
    Y_centered = Y - Y.mean(axis=0)
    pbv = Y_centered[:,0] - Y_centered[:,1]
    return pbv.astype(np.float32)


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

# ===== AUGMENTATION FUNCTIONS =====
AUGMENTATION = {
    "enabled": True,
    "p_brightness": 0.25,
    "p_contrast": 0.25,
    "p_noise": 0.25,
    "p_color_shift": 0.25,
    "p_motion_blur": 0.25,
    "p_jpeg": 0.25,
    "p_gamma": 0.25,
    "p_occlusion": 0.1,
    "p_crop": 0.05,
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

def safe_clip(arr, min_val=0, max_val=255):
    return np.clip(arr, min_val, max_val)

def augment_brightness(roi, cfg):
    try:
        factor = 1.0 + np.random.uniform(-cfg["brightness"], cfg["brightness"])
        return safe_clip(roi * factor)
    except Exception:
        return roi

def augment_contrast(roi, cfg):
    try:
        if roi.size == 0:
            return roi
        mean = np.mean(roi, axis=(0, 1), keepdims=True)
        factor = 1.0 + np.random.uniform(-cfg["contrast"], cfg["contrast"])
        return safe_clip((roi - mean) * factor + mean)
    except Exception:
        return roi

def augment_gaussian_noise(roi, cfg):
    try:
        noise = np.random.normal(0, cfg["gaussian_noise"], roi.shape)
        return safe_clip(roi + noise)
    except Exception:
        return roi

def augment_color_shift(roi, cfg):
    try:
        if len(roi.shape) != 3 or roi.shape[2] != 3:
            return roi
        shift = np.random.randint(-cfg["color_shift"], cfg["color_shift"] + 1, size=(1, 1, 3))
        return safe_clip(roi + shift)
    except Exception:
        return roi

def augment_motion_blur(roi, cfg):
    try:
        if roi.size == 0:
            return roi
        
        ksize = 3 if cfg["motion_blur_max_ksize"] < 5 else np.random.choice([3, 5])
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if np.random.rand() < 0.5:
            kernel[ksize//2, :] = 1.0
        else:
            kernel[:, ksize//2] = 1.0
        kernel /= ksize
        
        roi_uint8 = safe_clip(roi).astype(np.uint8)
        blurred = cv2.filter2D(roi_uint8, -1, kernel)
        return blurred.astype(np.float64)
    except Exception:
        return roi

def augment_frame(roi, aug_cfg):
    """Apply random augmentations to frame"""
    if not aug_cfg or not aug_cfg.get("enabled", False) or roi.size == 0:
        return safe_clip(roi).astype(np.uint8)
    
    try:
        roi = roi.astype(np.float64)
        
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
        
        np.random.shuffle(augmentations)
        for func in augmentations:
            roi = func(roi)
        
        return safe_clip(roi).astype(np.uint8)
    except Exception:
        return safe_clip(roi).astype(np.uint8)

def extract_window_face_rppg_signals(frames, boxes, start, end, fps=30, augment=True, augment_chance=0.5):
    rgb_signals = []
    window_boxes = boxes[start:end]
    window_mask = [1 if b is not None else 0 for b in window_boxes]
    for f, b in zip(frames[start:end], window_boxes):
        if b is not None:
            mean_rgb = extract_face_rgb_mean(f, b)
        else:
            mean_rgb = [0.0, 0.0, 0.0]
        rgb_signals.append(mean_rgb)
    # 5 signals
    signals = {}
    signals['chrom'] = apply_signal_preprocessing(rppg_chrom(rgb_signals), fps)
    signals['pos']   = apply_signal_preprocessing(rppg_pos(rgb_signals), fps)
    signals['ica']   = apply_signal_preprocessing(rppg_pca(rgb_signals), fps)
    signals['green'] = apply_signal_preprocessing(rppg_green(rgb_signals), fps)
    signals['pbv']   = apply_signal_preprocessing(rppg_pbv(rgb_signals), fps)
    # Optional: augmentation
    if augment:
        if random.random() < augment_chance: # CHANGE TO BE LOWER FOR FAKE, HIGHER FOR REAL
            augs = ['noise', 'amplitude', 'frequency', 'phase']
            aug = random.choice(augs)
            aug_mask = np.array(window_mask)
            for k in signals:
                signals[k] = augment_signal(signals[k], aug, mask=aug_mask)
    # Pad signals
    for k in signals:
        signals[k], _ = pad_mask(signals[k], WINDOW_SIZE)
    window_mask_padded = pad_mask(window_mask, WINDOW_SIZE)[1]
    return signals, window_mask_padded

# ============= VIDEO WINDOW EXTRACTION & QC ==================

def segment_windows(nframes, win=WINDOW_SIZE, hop=HOP_SIZE):
    """Generate sliding windows over video frames"""
    return [(s, min(s + win, nframes)) for s in range(0, max(nframes - win + 1, 1), hop)]

def process_video(video_path, augment_chance, augment=True):
    cap = None
    frames = []
    face_net = None
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0

        face_net = load_face_net(FACE_PROTO, FACE_MODEL)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        cap = None

        # No interpolation, keep raw frames (possibly None)
        boxes_seq = [detect_faces(frame, face_net) if frame is not None else [] for frame in frames]

        if len(frames) < MIN_REAL_FRAMES:
            return None
        tracks = robust_track_faces(boxes_seq)
        windows_all = []
        for tid, track in tracks.items():
            per_frame_boxes = [None] * len(frames)
            for fidx, box in track:
                per_frame_boxes[fidx] = box
            for start, end in segment_windows(len(frames)):
                if sum(1 for b in per_frame_boxes[start:end] if b is not None) < 2:
                    continue
                signals, window_mask = extract_window_face_rppg_signals(
                    frames, per_frame_boxes, start, end, fps=fps, augment=augment, augment_chance=augment_chance
                )
                if window_mask.sum() < MIN_REAL_FRAMES:
                    continue
                windows_all.append({
                    'signals': signals,   # dict: chrom/pos/ica/green/pbv (all [WINDOW_SIZE])
                    'window_mask': window_mask  # [WINDOW_SIZE]
                })
        return windows_all

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return None

    finally:
        if cap is not None:
            cap.release()
        try:
            del frames, face_net
        except Exception:
            pass
        gc.collect()

# ============== BATCH STORAGE =======================
def save_npz_batch(batch_data, out_path, label):
    # batch_data: dict {video: list of window dicts}
    X = []  # signals, shape: [N, 5, WINDOW_SIZE]
    M = []  # window masks, shape: [N, WINDOW_SIZE]
    vids = [] # for reference, which video
    win_idxs = [] # window idx per video
    for vid, windows in batch_data.items():
        for i, win in enumerate(windows):
            sigmat = np.stack([win['signals'][k] for k in ['chrom','pos','ica','green','pbv']], axis=0)
            X.append(sigmat)
            M.append(win['window_mask'])
            vids.append(vid)
            win_idxs.append(i)
    X = np.stack(X, axis=0)  # [N, 5, WINDOW_SIZE]
    M = np.stack(M, axis=0)
    np.savez_compressed(
        out_path, signals=X, masks=M, videos=np.array(vids), win_idxs=np.array(win_idxs), label=label
    )

# =================== MAIN MULTIPROC/PROCESSING ===================

def get_video_files(input_arg, exts=('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
    if input_arg.endswith('.txt'):
        with open(input_arg, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        return [os.path.join(input_arg, f) for f in os.listdir(input_arg)
                if f.lower().endswith(exts)]

def get_next_batch_idx(output_dir, label):
    batch_files = [f for f in os.listdir(output_dir) if f.startswith(f"{label}_batch") and f.endswith(".npz")]
    max_idx = -1
    for fname in batch_files:
        m = re.match(rf"{label}_batch(\d+)\.npz", fname)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--augment_chance', type=float, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    videos = get_video_files(args.input)
    print(f"Found {len(videos)} videos in {args.input}")

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
    random.shuffle(videos)

    logger.info(f"Processing {len(videos)} new videos (label: {args.label})")

    batch_idx = batch_idx = get_next_batch_idx(args.output, args.label)
    n_in_batch = 0
    batch_data = {}
    just_batched_videos = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as exe:
        future_to_vid = {exe.submit(process_video, v, args.augment_chance): v for v in videos}
        for fut in tqdm(as_completed(future_to_vid), total=len(future_to_vid), desc='Videos'):
            vid = future_to_vid[fut]
            try:
                windows = fut.result()
            except Exception as e:
                print(f"Error processing {vid}: {e}")
                windows = None

            if windows and len(windows) > 0:
                batch_data[vid] = windows
                n_in_batch += 1
                just_batched_videos.append(os.path.basename(vid))
            else:
                # Log as failed
                with open(failure_log, 'a') as f:
                    f.write(os.path.basename(vid) + '\n')

            if n_in_batch >= args.batch_size:
                out_path = os.path.join(args.output, f'{args.label}_batch{batch_idx}.npz')
                save_npz_batch(batch_data, out_path, args.label)
                print(f"Saved batch {batch_idx} ({n_in_batch} videos) to {out_path}")
                with open(log_file, 'a') as f:
                    for v in just_batched_videos:
                        f.write(v + '\n')
                just_batched_videos = []
                batch_data = {}
                n_in_batch = 0
                batch_idx += 1

        # Save the final partial batch after loop ends
        if batch_data:
            out_path = os.path.join(args.output, f'{args.label}_batch{batch_idx}.npz')
            save_npz_batch(batch_data, out_path, args.label)
            print(f"Saved final batch {batch_idx} ({n_in_batch} videos) to {out_path}")
            with open(log_file, 'a') as f:
                for v in just_batched_videos:
                    f.write(v + '\n')

if __name__ == '__main__':
    main()