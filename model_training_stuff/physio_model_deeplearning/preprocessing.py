import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from scipy.signal import detrend, butter, filtfilt, periodogram
from scipy import linalg
import logging
import warnings
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import re
import threading
from collections import OrderedDict
import uuid
import scipy.stats
import random
import fcntl  # For Unix/Linux
import msvcrt  # For Windows
import platform
import time

os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Thread-safe MediaPipe instances using thread-local storage
_thread_local = threading.local()

def get_mediapipe_instance():
    """Get thread-safe MediaPipe instance using thread-local storage"""
    if not hasattr(_thread_local, 'mp_face_mesh'):
        try:
            import mediapipe as mp
            _thread_local.mp_face_mesh_solution = mp.solutions.face_mesh
            _thread_local.mp_face_mesh = _thread_local.mp_face_mesh_solution.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,  # Optimize for single face
                refine_landmarks=False,  # Speed up processing
                min_detection_confidence=0.5
            )
        except ImportError:
            logger.warning("MediaPipe not available, landmark detection will be skipped")
            _thread_local.mp_face_mesh = None
            _thread_local.mp_face_mesh_solution = None
    return getattr(_thread_local, 'mp_face_mesh', None), getattr(_thread_local, 'mp_face_mesh_solution', None)

# Thread-safe FFT cache using OrderedDict for proper LRU behavior
_cache_lock = threading.Lock()
_fft_cache = OrderedDict()
_max_cache_size = 1000

def cached_fft(signal, cache_key=None):
    """Thread-safe FFT computation with proper LRU caching"""
    if cache_key is None:
        cache_key = hash(signal.tobytes())
    
    with _cache_lock:
        if cache_key in _fft_cache:
            # Move to end (most recently used)
            _fft_cache.move_to_end(cache_key)
            return _fft_cache[cache_key]
        
        # Compute FFT
        fft_result = np.fft.fft(signal)
        
        # Add to cache
        _fft_cache[cache_key] = fft_result
        
        # Maintain cache size by removing oldest entries
        while len(_fft_cache) > _max_cache_size:
            _fft_cache.popitem(last=False)  # Remove oldest (FIFO)
    
    return fft_result

def file_lock(file_path, timeout=30):
    """Cross-platform file locking context manager"""
    import tempfile
    lock_file = f"{file_path}.lock"
    
    class FileLock:
        def __init__(self, lock_path, timeout):
            self.lock_path = lock_path
            self.timeout = timeout
            self.lock_file = None
            
        def __enter__(self):
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    self.lock_file = open(self.lock_path, 'w')
                    if platform.system() == 'Windows':
                        msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except (IOError, OSError):
                    if self.lock_file:
                        self.lock_file.close()
                    time.sleep(0.1)
            raise TimeoutError(f"Could not acquire lock for {self.lock_path}")
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.lock_file:
                if platform.system() == 'Windows':
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
                try:
                    os.unlink(self.lock_path)
                except:
                    pass
    
    return FileLock(lock_file, timeout)

# =================== HELPER FUNCTIONS ==========================
def validate_landmarks(landmarks, expected_count=468):
    """Validate MediaPipe landmarks count and structure"""
    if landmarks is None:
        return False
    if len(landmarks) != expected_count:
        logger.warning(f"Expected {expected_count} landmarks, got {len(landmarks)}")
        return False
    return True

def get_face_landmarks(frame, box):
    """Use facial landmark detector with proper error handling and validation"""
    face_mesh, _ = get_mediapipe_instance()
    if face_mesh is None:
        return None
        
    # Validate input frame
    if frame is None or len(frame.shape) != 3 or frame.shape[2] not in [3, 4]:
        logger.warning("Frame is not in expected BGR/RGB format")
        return None
        
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]
    
    # Bounds checking with validation
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    
    # Ensure ROI is large enough for processing
    if w < 50 or h < 50:
        logger.warning(f"Face ROI too small: {w}x{h}")
        return None
    
    roi_frame = frame[y:y+h, x:x+w]
    
    # Handle different channel formats
    if frame.shape[2] == 4:  # RGBA
        roi_frame = roi_frame[:, :, :3]  # Drop alpha channel
    
    try:
        rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
    except Exception as e:
        logger.warning(f"MediaPipe processing failed: {e}")
        return None
    
    if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
        return None
        
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Validate landmark count
    if not validate_landmarks(landmarks):
        return None
    
    # Convert landmarks to pixel coordinates with bounds checking
    valid_landmarks = []
    for l in landmarks:
        lx, ly = int(l.x * w), int(l.y * h)
        # Ensure landmarks are within bounds
        if 0 <= lx < w and 0 <= ly < h:
            valid_landmarks.append((lx, ly))
        else:
            # Use clamped coordinates instead of (0,0) fallback
            lx = max(0, min(lx, w-1))
            ly = max(0, min(ly, h-1))
            valid_landmarks.append((lx, ly))
            
    return valid_landmarks

def extract_multi_roi_signals(frame, box, landmarks, rois=['left_cheek', 'right_cheek', 'forehead', 'chin', 'nose']):
    """Extract mean RGB signals for each ROI with improved error handling"""
    if frame is None or len(frame.shape) != 3 or frame.shape[2] < 3:
        logger.warning("Frame is not in expected BGR format")
        return {}
        
    if landmarks is None or len(landmarks) == 0:
        return {}
        
    # Validate landmarks count for MediaPipe face mesh
    if len(landmarks) < 468:
        logger.warning(f"Insufficient landmarks: {len(landmarks)} < 468")
        return {}
        
    h, w = box[3], box[2]
    
    # ROI landmark indices for MediaPipe 468-point face mesh
    roi_points = {
        'left_cheek':  [234, 93, 132, 58, 172, 136, 150, 149],
        'right_cheek': [454, 323, 361, 288, 397, 365, 379, 378],
        'forehead':    [10, 338, 297, 332, 284, 251, 21, 54],
        'chin':        [152, 148, 176, 149, 150, 136, 172, 58],
        'nose':        [2, 98, 327, 195, 3, 51, 48, 115]
    }
    
    rgb_means = {}
    for roi in rois:
        if roi not in roi_points:
            continue
            
        # Validate landmark indices before accessing
        valid_indices = [i for i in roi_points[roi] if i < len(landmarks)]
        if len(valid_indices) < 3:  # Need at least 3 points for a polygon
            logger.warning(f"Insufficient valid landmarks for ROI {roi}: {len(valid_indices)}")
            continue
            
        try:
            pts = np.array([landmarks[i] for i in valid_indices], dtype=np.int32)
            mask = np.zeros((h, w), np.uint8)
            
            if pts.shape[0] > 2:
                cv2.fillPoly(mask, [pts], 1)
                
                # Add bounds checking for frame access
                x, y = box[0], box[1]
                frame_h, frame_w = frame.shape[:2]
                
                # Ensure we don't go out of bounds
                y_end = min(y + h, frame_h)
                x_end = min(x + w, frame_w)
                
                if y < frame_h and x < frame_w and y_end > y and x_end > x:
                    roi_region = frame[y:y_end, x:x_end]
                    
                    # Adjust mask size if needed
                    if roi_region.shape[:2] != mask.shape:
                        mask_h, mask_w = mask.shape
                        roi_h, roi_w = roi_region.shape[:2]
                        # Crop mask to match ROI region
                        mask = mask[:min(mask_h, roi_h), :min(mask_w, roi_w)]
                    
                    # Extract mean RGB values with channel validation
                    roi_means_list = []
                    n_channels = min(3, roi_region.shape[2])  # Process up to 3 channels
                    
                    for c in range(n_channels):
                        masked_pixels = roi_region[:, :, c][mask == 1]
                        if masked_pixels.size > 0:
                            roi_means_list.append(float(np.mean(masked_pixels)))
                        else:
                            roi_means_list.append(0.0)
                    
                    # Ensure exactly 3 values (RGB)
                    while len(roi_means_list) < 3:
                        roi_means_list.append(0.0)
                    
                    rgb_means[roi] = roi_means_list[:3]
                    
        except Exception as e:
            logger.warning(f"Error extracting ROI {roi}: {e}")
            continue
    
    return rgb_means

def extract_signals_for_track(frames, per_frame_boxes, window_size, start, end, fps=30):
    """
    Extract multi-ROI physiological signals for one track window.
    OPTIMIZED: Only extract core rPPG methods with frame interpolation support.
    """
    # Extract global RGB with frame interpolation
    window_frames = frames[start:end]
    window_boxes = per_frame_boxes[start:end]

    # Global face signal
    rgb_signal, face_mask = extract_rgb_signal_track(window_frames, window_boxes)

    # Multi-ROI signals with improved error handling
    roi_signals = {}  # {roi: rgb_signal}
    roi_names = ['left_cheek', 'right_cheek', 'forehead', 'chin', 'nose']
    
    for roi in roi_names:
        roi_rgb_signal = []
        for frame, box in zip(window_frames, window_boxes):
            if box is not None and frame is not None:
                landmarks = get_face_landmarks(frame, box)
                if landmarks:
                    roi_means = extract_multi_roi_signals(frame, box, landmarks, rois=[roi])
                    vals = roi_means.get(roi, [0.0, 0.0, 0.0])
                    roi_rgb_signal.append(vals)
                else:
                    roi_rgb_signal.append([0.0, 0.0, 0.0])
            else:
                roi_rgb_signal.append([0.0, 0.0, 0.0])
        roi_signals[roi] = np.array(roi_rgb_signal, dtype=np.float32)

    # Extract physio signals - OPTIMIZED VERSION
    signals = {}
    # Multi-ROI signals
    signals['multi_roi'] = {roi: extract_core_physio_signals(roi_signals[roi], fps) for roi in roi_names}
    signals['face_mask'] = face_mask  # for QC
    
    return signals

def pad_and_mask(array, window_size, pad_mode='zero'):
    """Pad array (1D or 2D) to window_size and return mask with validation"""
    array = np.array(array)
    if array.size == 0:
        # Handle empty array case
        if array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1):
            padded = np.zeros(window_size, dtype=np.float32)
        else:
            # Assume 2D array, use last known shape or default
            n_features = array.shape[1] if array.ndim == 2 else 1
            padded = np.zeros((window_size, n_features), dtype=np.float32)
        mask = np.zeros(window_size, dtype=np.float32)
        return padded, mask
    
    L = array.shape[0]
    if L >= window_size:
        return array[:window_size].astype(np.float32), np.ones(window_size, dtype=np.float32)
    
    pad_len = window_size - L
    
    if array.ndim == 1:
        if L > 0:
            if pad_mode == 'repeat_last':
                pad = np.repeat(array[-1], pad_len)
            else:  # Default to zero padding
                pad = np.zeros(pad_len, dtype=array.dtype)
            padded = np.concatenate([array, pad], axis=0)
        else:
            padded = np.zeros(window_size, dtype=array.dtype)
    else:  # 2D array
        if L > 0:
            if pad_mode == 'repeat_last':
                pad = np.repeat(array[-1:], pad_len, axis=0)
            else:  # Default to zero padding
                pad = np.zeros((pad_len, array.shape[1]), dtype=array.dtype)
            padded = np.concatenate([array, pad], axis=0)
        else:
            n_features = array.shape[1] if array.shape[1] > 0 else 1
            padded = np.zeros((window_size, n_features), dtype=array.dtype)
    
    # Create mask (1=real, 0=pad)
    mask = np.concatenate([np.ones(L, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)], axis=0)
    
    return padded.astype(np.float32), mask

# ========== OPTIMIZED rPPG EXTRACTION METHODS ==========

def rppg_chrom(rgb_signal):
    """CHROM method for rPPG extraction with improved error handling"""
    S = np.array(rgb_signal, dtype=np.float32)
    min_length = 10
    
    if S.size == 0 or S.shape[0] < min_length:
        return np.zeros(max(S.shape[0], 1), dtype=np.float32)
    
    # Ensure we have 3 channels
    if S.ndim == 1 or S.shape[1] < 3:
        return np.zeros(S.shape[0], dtype=np.float32)
    
    # Better division by zero protection with numerical stability
    S_mean = np.mean(S, axis=0)
    S_std = np.std(S, axis=0)
    
    # Prevent division by zero and very small values
    S_std = np.where(S_std < 1e-6, 1e-6, S_std)
    S_mean = np.where(np.abs(S_mean) < 1e-6, 1e-6, S_mean)
    
    # Standardize
    S_norm = (S - S_mean) / S_std
    
    # CHROM combination: h = G - R, s = G + R - 2B
    try:
        h = S_norm[:, 1] - S_norm[:, 0]  # G - R
        s = S_norm[:, 1] + S_norm[:, 0] - 2 * S_norm[:, 2]  # G + R - 2B
        return (h + s).astype(np.float32)
    except IndexError:
        logger.warning("CHROM: Insufficient channels in RGB signal")
        return np.zeros(S.shape[0], dtype=np.float32)

def rppg_pos(rgb_signal):
    """POS (Plane Orthogonal to Skin) method with improved stability"""
    S = np.array(rgb_signal, dtype=np.float32)
    min_length = 10
    
    if S.size == 0 or S.shape[0] < min_length:
        return np.zeros(max(S.shape[0], 1), dtype=np.float32)
    
    # Ensure we have 3 channels
    if S.ndim == 1 or S.shape[1] < 3:
        return np.zeros(S.shape[0], dtype=np.float32)
    
    try:
        # Spatial averaging with protection against zero values
        S_mean = np.mean(S, axis=0)
        S_mean = np.where(np.abs(S_mean) < 1e-6, 1e-6, S_mean)
        
        # Normalize
        S_norm = S / S_mean
        
        # POS combination matrix
        P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
        Y = np.dot(S_norm, P.T)
        
        # Compute optimal projection
        if Y.shape[0] > 1 and Y.shape[1] >= 2:
            std_y = np.std(Y, axis=0)
            # Better division by zero protection
            std_y = np.where(std_y < 1e-6, 1e-6, std_y)
            Y_norm = Y / std_y
            
            # Alpha tuning with protection
            alpha_denom = np.std(Y_norm[:, 1])
            alpha_num = np.std(Y_norm[:, 0])
            alpha = alpha_num / max(alpha_denom, 1e-6)
            
            # Final signal
            rppg = Y_norm[:, 0] - alpha * Y_norm[:, 1]
        else:
            rppg = Y[:, 0] if Y.shape[1] > 0 else np.zeros(Y.shape[0])
        
        return rppg.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"POS method failed: {e}")
        return np.zeros(S.shape[0], dtype=np.float32)

def rppg_ica(rgb_signal):
    """ICA-based rPPG extraction with consistent fallback handling"""
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        logger.warning("scikit-learn not available, falling back to CHROM")
        return rppg_chrom(rgb_signal)
    
    S = np.array(rgb_signal, dtype=np.float32)
    min_length = 30  # ICA needs more samples
    
    if S.size == 0 or S.shape[0] < min_length:
        return rppg_chrom(rgb_signal)  # Consistent fallback
    
    # Ensure we have 3 channels
    if S.ndim == 1 or S.shape[1] < 3:
        return rppg_chrom(rgb_signal)  # Consistent fallback
    
    try:
        # Center the signal
        rgb_centered = S - np.mean(S, axis=0)
        
        # Check for constant signals
        if np.allclose(rgb_centered, 0, atol=1e-6):
            return rppg_chrom(rgb_signal)
        
        # Apply ICA with better parameters
        ica = FastICA(n_components=3, random_state=42, max_iter=1000, tol=1e-4)
        sources = ica.fit_transform(rgb_centered)
        
        # Select component with highest power in HR frequency range
        best_source = None
        max_power = -1
        
        for i in range(min(sources.shape[1], 3)):  # Process up to 3 components
            source = sources[:, i]
            
            # Simple frequency analysis using cached FFT
            cache_key = f"ica_{hash(source.tobytes())}"
            fft = cached_fft(source, cache_key)
            freqs = np.fft.fftfreq(len(source), d=1/30.0)  # Assume 30 fps
            
            # HR range: 0.7-4 Hz (42-240 BPM)
            hr_mask = (np.abs(freqs) >= 0.7) & (np.abs(freqs) <= 4.0)
            if np.any(hr_mask):
                hr_power = np.sum(np.abs(fft[hr_mask])**2)
                
                if hr_power > max_power:
                    max_power = hr_power
                    best_source = source
        
        result = best_source if best_source is not None else sources[:, 0]
        return result.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"ICA failed: {e}, falling back to CHROM")
        return rppg_chrom(rgb_signal)  # Consistent fallback

# ========== SIGNAL AUGMENTATION FOR DEEP LEARNING ==========

def augment_physiological_signal(signal, aug_type='none', strength=0.1, mask=None):
    """Apply physiologically plausible augmentations with proper broadcasting"""
    if aug_type == 'none' or signal.size == 0:
        return signal.astype(np.float32)
    
    augmented = signal.copy().astype(np.float32)
    
    # Apply augmentation only to real (non-padded) portions
    if mask is not None:
        real_indices = mask > 0.5  # Real data where mask is 1
        if not np.any(real_indices):
            return augmented  # No real data to augment
    else:
        real_indices = np.ones(len(signal), dtype=bool)  # Augment all if no mask
    
    real_length = np.sum(real_indices)
    if real_length == 0:
        return augmented
    
    try:
        if aug_type == 'noise':
            # Add colored noise (more realistic than white noise)
            noise = np.random.randn(real_length).astype(np.float32)
            # Apply 1/f filter to make pink noise
            if real_length > 1:
                fft = np.fft.fft(noise)
                freqs = np.fft.fftfreq(real_length)
                nonzero_mask = freqs != 0
                if np.any(nonzero_mask):
                    fft[nonzero_mask] = fft[nonzero_mask] / np.sqrt(np.abs(freqs[nonzero_mask]))
                noise = np.real(np.fft.ifft(fft)).astype(np.float32)
                
                # Proper broadcasting: apply noise only to real portions
                real_signal = augmented[real_indices]
                if len(real_signal) > 0:
                    signal_std = np.std(real_signal)
                    augmented[real_indices] = real_signal + strength * noise * signal_std
            
        elif aug_type == 'amplitude':
            # Random amplitude modulation - fixed broadcasting issue
            if real_length > 0:
                modulation = 1 + strength * np.sin(
                    2 * np.pi * np.random.rand() * np.arange(real_length, dtype=np.float32) / real_length
                )
                real_signal = augmented[real_indices]
                augmented[real_indices] = real_signal * modulation
            
        elif aug_type == 'frequency':
            # Slight frequency shift (simulating HR variations)
            real_signal = augmented[real_indices]
            if real_length > 1:
                shift_factor = 1 + np.random.uniform(-strength, strength)
                indices = np.arange(real_length, dtype=np.float32)
                new_indices = indices * shift_factor
                
                # Ensure indices are within bounds
                valid_mask = new_indices < real_length
                if np.any(valid_mask):
                    valid_new_indices = new_indices[valid_mask]
                    valid_signal = real_signal[:len(valid_new_indices)]
                    
                    interpolated = np.interp(indices, valid_new_indices, valid_signal)
                    augmented[real_indices] = interpolated.astype(np.float32)
            
        elif aug_type == 'phase':
            # Random phase shift
            real_signal = augmented[real_indices]
            if real_length > 1:
                shift = int(strength * real_length * np.random.rand())
                augmented[real_indices] = np.roll(real_signal, shift)
                
    except Exception as e:
        logger.warning(f"Signal augmentation failed: {e}")
        return signal.astype(np.float32)  # Return original on failure
        
    return augmented

# ========== OPTIMIZED PREPROCESSING FOR DEEP LEARNING ==========

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

def extract_core_physio_signals(rgb_signal, fs=30):
    """
    Extract ONLY core physiological signals for deep learning.
    Returns exactly 4 signals with consistent dimensions.
    """
    signals = {}
    
    # Ensure input is proper numpy array
    rgb_signal = np.array(rgb_signal, dtype=np.float32)
    
    if rgb_signal.size == 0:
        # Return empty signals with consistent structure
        empty_signal = np.array([], dtype=np.float32)
        signals['chrom'] = empty_signal
        signals['pos'] = empty_signal
        signals['ica'] = empty_signal
        return signals
    
    # Core rPPG methods only
    raw_chrom = rppg_chrom(rgb_signal)
    raw_pos = rppg_pos(rgb_signal)
    raw_ica = rppg_ica(rgb_signal)
    
    # Apply light preprocessing to each signal
    signals['chrom'] = apply_signal_preprocessing(raw_chrom, fs)
    signals['pos'] = apply_signal_preprocessing(raw_pos, fs)
    signals['ica'] = apply_signal_preprocessing(raw_ica, fs)
    
    return signals

def extract_ml_features(signals, fs=30):
    """
    Compute ML-friendly summary features for a window.
    signals: dict, structure like signals['multi_roi'][roi][method] = 1D np.array
    Returns: dict of flat features.
    """
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
            # Avoid division by zero
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
            # Center the signal
            centered = sig - np.mean(sig)
            # Count zero crossings
            crossings = np.sum(np.diff(np.sign(centered)) != 0)
            return float(crossings / len(sig))
        except:
            return 0.0

    feats = {}
    
    # Extract features from multi-ROI signals
    for roi in signals.get('multi_roi', {}):
        for method in signals['multi_roi'][roi]:
            sig = np.array(signals['multi_roi'][roi][method], dtype=np.float32)
            
            # Use window_mask to get only real (non-padded) data
            if 'window_mask' in signals:
                mask = np.array(signals['window_mask'])
                real_idx = mask > 0.5
                if np.any(real_idx):
                    sig = sig[real_idx]
            
            # Skip if insufficient data
            if len(sig) < 10:
                continue
            
            # Remove any invalid values
            sig = sig[np.isfinite(sig)]
            if len(sig) < 10:
                continue

            prefix = f'{roi}_{method}'
            
            # Basic statistical features
            try:
                feats[f'{prefix}_mean'] = float(np.mean(sig))
                feats[f'{prefix}_std'] = float(np.std(sig))
                feats[f'{prefix}_var'] = float(np.var(sig))
                feats[f'{prefix}_min'] = float(np.min(sig))
                feats[f'{prefix}_max'] = float(np.max(sig))
                feats[f'{prefix}_median'] = float(np.median(sig))
                feats[f'{prefix}_range'] = float(np.max(sig) - np.min(sig))
                
                # Percentiles
                feats[f'{prefix}_q25'] = float(np.percentile(sig, 25))
                feats[f'{prefix}_q75'] = float(np.percentile(sig, 75))
                feats[f'{prefix}_iqr'] = float(np.percentile(sig, 75) - np.percentile(sig, 25))
                
                # Higher order moments
                feats[f'{prefix}_skew'] = float(scipy.stats.skew(sig))
                feats[f'{prefix}_kurt'] = float(scipy.stats.kurtosis(sig))
                
                # Frequency domain features
                feats[f'{prefix}_bandpower'] = _bandpower(sig, fs)
                feats[f'{prefix}_spectral_centroid'] = _spectral_centroid(sig, fs)
                
                # Time domain features
                feats[f'{prefix}_zero_crossing_rate'] = _zero_crossing_rate(sig)
                feats[f'{prefix}_energy'] = float(np.sum(sig**2))
                feats[f'{prefix}_rms'] = float(np.sqrt(np.mean(sig**2)))
                
                # Signal quality indicators
                feats[f'{prefix}_snr_estimate'] = float(np.var(sig) / max(np.var(np.diff(sig)), 1e-10))
                
            except Exception as e:
                logger.warning(f"Error computing features for {prefix}: {e}")
                continue
    
    # Add global features if available
    if 'face_mask' in signals:
        face_mask = np.array(signals['face_mask'])
        feats['face_detection_rate'] = float(np.mean(face_mask))
    
    if 'window_mask' in signals:
        window_mask = np.array(signals['window_mask'])
        feats['data_completeness'] = float(np.mean(window_mask))
        feats['valid_frames'] = float(np.sum(window_mask))
    
    return feats

# ============= MAIN PREPROCESSING FUNCTIONS ==================
def process_one_video(args):
    """Process single video with improved error handling"""
    video_file, video_dir, window_size, hop_size, augment = args
    video_path = os.path.join(video_dir, video_file)
    try:
        windows = preprocess_video_for_dl(video_path, window_size, hop_size, augment)
        return (video_file, windows)
    except Exception as e:
        logger.error(f"Error processing {video_file}: {e}")
        return (video_file, None)

def preprocess_video_for_dl(video_path, window_size=150, hop_size=75, augment=True):
    """
    Preprocess video for deep learning with FRAME INTERPOLATION for gap handling
    Returns raw signal windows with reduced feature set
    """
    try:
        # Validate video file exists before processing
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return None
        
        # Initialize face detection
        face_net = get_face_net()
        if face_net is None:
            logger.error(f"Face detection model not loaded")
            return None
        
        # Open video with better error handling using context manager
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        try:
            frames = []
            all_boxes = []
            frame_indices = []  # Track actual frame positions for debugging
            
            # Better FPS handling
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or np.isnan(fps):
                logger.warning(f"Invalid FPS detected: {fps}, defaulting to 30")
                fps = 30.0
            
            # Read frames with FRAME INTERPOLATION for gap handling
            frame_count = 0
            max_frames = 10000  # Prevent infinite loops
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            while frame_count < max_frames and consecutive_failures < max_consecutive_failures:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Validate frame with INTERPOLATION strategy
                if frame is None or frame.size == 0:
                    consecutive_failures += 1
                    logger.warning(f"Corrupted frame at index {frame_count}")
                    
                    # FRAME INTERPOLATION: Use last valid frame if available
                    if len(frames) > 0:
                        # Interpolate by repeating last valid frame
                        interpolated_frame = frames[-1].copy()
                        interpolated_boxes = all_boxes[-1] if all_boxes else []
                        
                        frames.append(interpolated_frame)
                        all_boxes.append(interpolated_boxes)
                        frame_indices.append(frame_count)
                        
                        logger.info(f"Interpolated frame at index {frame_count} using last valid frame")
                    else:
                        logger.warning(f"No previous frame available for interpolation at index {frame_count}")
                        continue
                else:
                    # Valid frame - reset failure counter
                    consecutive_failures = 0
                    
                    # Resize frame with aspect ratio preservation
                    try:
                        original_h, original_w = frame.shape[:2]
                        target_w, target_h = 640, 360
                        
                        # Calculate scaling to maintain aspect ratio
                        scale = min(target_w / original_w, target_h / original_h)
                        new_w = int(original_w * scale)
                        new_h = int(original_h * scale)
                        
                        frame = cv2.resize(frame, (new_w, new_h))
                        
                        # Pad to target size if needed
                        if new_w != target_w or new_h != target_h:
                            pad_w = (target_w - new_w) // 2
                            pad_h = (target_h - new_h) // 2
                            frame = cv2.copyMakeBorder(frame, pad_h, target_h - new_h - pad_h,
                                                     pad_w, target_w - new_w - pad_w,
                                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        
                    except Exception as e:
                        logger.warning(f"Frame resize failed at index {frame_count}: {e}")
                        continue
                    
                    # Detect faces
                    boxes = detect_faces_dnn(frame, face_net)
                    
                    frames.append(frame)
                    all_boxes.append(boxes)
                    frame_indices.append(frame_count)
                
                frame_count += 1
            
        finally:
            cap.release()  # Ensure resource cleanup
        
        # Validate minimum video length
        if len(frames) < window_size:
            logger.warning(f"Video too short after processing: {len(frames)} frames < {window_size}")
            return None
        
        # Quality check: ensure we didn't lose too many frames to interpolation
        interpolation_rate = consecutive_failures / max(frame_count, 1)
        if interpolation_rate > 0.3:  # More than 30% interpolated frames
            logger.warning(f"High interpolation rate ({interpolation_rate:.1%}) in video {video_path}")
        
        # Track faces with better error handling
        try:
            tracks = robust_track_faces(all_boxes)
        except Exception as e:
            logger.error(f"Face tracking failed: {e}")
            return None
            
        if not tracks:
            logger.warning(f"No valid face tracks found")
            return None
        
        all_windows = []
        for track_id, track in tracks.items():
            per_frame_boxes = [None] * len(frames)
            for (frame_idx, box) in track:
                if frame_idx < len(frames):
                    per_frame_boxes[frame_idx] = box

            # Sliding window extraction
            n = len(frames)
            starts = list(range(0, max(n - window_size + 1, 1), hop_size))
            if n > window_size and (n - window_size) % hop_size != 0:
                starts.append(n - window_size)
            elif n < window_size:
                starts = [0]  # Only one window: pad all

            for start in starts:
                end = start + window_size
                if end > n:
                    # Last window is short, needs padding
                    actual_end = n
                else:
                    actual_end = end

                signals = extract_signals_for_track(frames, per_frame_boxes, window_size, start, actual_end, fps=fps)

                # Use zero padding consistently
                padded_face_mask, mask_mask = pad_and_mask(signals['face_mask'], window_size, pad_mode='zero')
                
                for roi in signals['multi_roi']:
                    for key in signals['multi_roi'][roi]:
                        arr = np.array(signals['multi_roi'][roi][key], dtype=np.float32)
                        signals['multi_roi'][roi][key], _ = pad_and_mask(arr, window_size, pad_mode='zero')
                
                # Use 60% threshold for minimum real frames
                min_real_frames = int(0.6 * window_size)
                if np.sum(mask_mask) < min_real_frames:
                    continue  # Skip low-information (mostly-padded) window

                # Overwrite with padded face mask and add window mask
                signals['face_mask'] = padded_face_mask
                signals['window_mask'] = mask_mask  # Save for model

                # Quality checks on real (non-padded) portion only
                real_len = actual_end - start
                std_threshold = 1e-3
                
                # Check ROI signal quality (do this for each window)
                try:
                    roi_stds = []
                    for roi in signals['multi_roi']:
                        if 'chrom' in signals['multi_roi'][roi]:
                            roi_std = np.std(signals['multi_roi'][roi]['chrom'][:real_len])
                            roi_stds.append(roi_std)
                    if not roi_stds or np.mean(roi_stds) < std_threshold:
                        continue
                except (KeyError, IndexError):
                    continue


                # Check ROI signal quality
                try:
                    roi_stds = []
                    for roi in signals['multi_roi']:
                        if 'chrom' in signals['multi_roi'][roi]:
                            roi_std = np.std(signals['multi_roi'][roi]['chrom'][:real_len])
                            roi_stds.append(roi_std)
                    
                    if not roi_stds or np.mean(roi_stds) < std_threshold:
                        continue
                except (KeyError, IndexError):
                    continue

                # Improved augmentation with proper masking
                if augment:
                    aug_types = ['noise', 'amplitude', 'frequency', 'phase']
                    aug_type = np.random.choice(aug_types)
                    mask = signals['window_mask']  # (shape: [window_size], 1=real, 0=pad)
                    
                    # Apply augmentation to multi-ROI signals
                    for roi in signals['multi_roi']:
                        if np.random.rand() < 0.5:
                            for key in signals['multi_roi'][roi]:
                                signals['multi_roi'][roi][key] = augment_physiological_signal(
                                    signals['multi_roi'][roi][key], 
                                    aug_type=aug_type, 
                                    strength=np.random.uniform(0.05, 0.15),
                                    mask=mask
                                )
                # Extract ML features for the window
                ml_feats = extract_ml_features(signals, fs=int(fps))

                # Add ML features to the signals dict
                signals['ml_features'] = ml_feats
                all_windows.append(signals)

        return all_windows
        
    except Exception as e:
        logger.error(f"Video preprocessing failed for {video_path}: {e}")
        return None

def sanitize_filename(filename):
    """Sanitize filename for HDF5 group names with process-safe unique ID"""
    # Remove file extension and invalid characters
    name = os.path.splitext(filename)[0]
    
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Remove any remaining problematic characters
    name = re.sub(r'[^\w\-_.]', '_', name)
    
    # Ensure name doesn't start with a number (HDF5 requirement)
    if name and name[0].isdigit():
        name = 'v_' + name
    
    # Ensure name is not empty
    if not name:
        name = 'unknown_video'
    
    # Limit length to prevent issues
    if len(name) > 200:
        name = name[:200]
    
    # Add process-specific and time-based unique identifier
    pid = os.getpid()
    thread_id = threading.get_ident()
    timestamp = int(time.time() * 1000000)  # microsecond precision
    unique_id = f"{pid}_{thread_id}_{timestamp}"
    
    # Ensure final name is not too long
    max_base_length = 240 - len(unique_id) - 1  # Leave room for underscore and unique_id
    if len(name) > max_base_length:
        name = name[:max_base_length]
    
    name = f"{name}_{unique_id}"
    
    return name

def merge_temp_hdf5_files(temp_files, output_path, label):
    """Merge temporary HDF5 files into final batch file"""
    try:
        with h5py.File(output_path, 'w') as output_f:
            output_f.attrs['dataset_label'] = label
            output_f.attrs['is_fake'] = 1 if 'fake' in label.lower() else 0
            
            for temp_file in temp_files:
                if not os.path.exists(temp_file):
                    continue
                    
                with h5py.File(temp_file, 'r') as temp_f:
                    for group_name in temp_f.keys():
                        temp_f.copy(group_name, output_f)
                        
    except Exception as e:
        logger.error(f"Failed to merge temp files: {e}")
        raise

def save_raw_signals_hdf5(output_path, video_windows_dict, label):
    """
    Save multi-ROI signals in HDF5 with proper file locking and race condition prevention.
    Combines fixes #5 (HDF5 thread safety) and #8 (race condition in group names).
    """
    try:
        # Use file locking to prevent race conditions (Fix #5)
        with file_lock(output_path):
            # Check if file exists and validate it first
            file_exists = os.path.exists(output_path)
            if file_exists:
                try:
                    with h5py.File(output_path, 'r') as test_f:
                        pass  # Just test if file is readable
                except Exception as e:
                    logger.warning(f"Existing file corrupted, recreating: {e}")
                    try:
                        os.remove(output_path)
                    except:
                        pass
                    file_exists = False
            
            # Open file with appropriate mode
            mode = 'a' if file_exists else 'w'
            
            with h5py.File(output_path, mode) as f:
                # Store dataset-level label only if new file
                if not file_exists:
                    f.attrs['dataset_label'] = label
                    f.attrs['is_fake'] = 1 if 'fake' in label.lower() else 0
                    f.attrs['feature_count_per_roi'] = 3
                    f.attrs['roi_count'] = 5
                
                for video_name, windows in video_windows_dict.items():
                    if not windows:
                        continue
                    
                    # Generate process-safe name (Fix #8)
                    safe_video_name = sanitize_filename(video_name)
                    
                    # Additional collision detection with exponential backoff and randomization
                    counter = 0
                    original_name = safe_video_name
                    max_attempts = 100
                    
                    while safe_video_name in f and counter < max_attempts:
                        # Add random component to avoid systematic collisions
                        random_suffix = f"{counter}_{random.randint(1000, 9999)}"
                        safe_video_name = f"{original_name}_{random_suffix}"
                        counter += 1
                        
                        # Small sleep to reduce collision probability
                        time.sleep(0.001)
                        
                        if counter >= max_attempts:
                            # Last resort: use timestamp with random number
                            emergency_name = f"emergency_{int(time.time()*1000000)}_{random.randint(10000, 99999)}"
                            safe_video_name = emergency_name
                            logger.warning(f"Used emergency group name: {safe_video_name}")
                            break
                    
                    try:
                        video_group = f.create_group(safe_video_name)
                    except ValueError as e:
                        if "name already exists" in str(e).lower():
                            # Final fallback with UUID
                            safe_video_name = f"{original_name}_{str(uuid.uuid4()).replace('-', '_')}"
                            video_group = f.create_group(safe_video_name)
                            logger.warning(f"Used UUID fallback group name: {safe_video_name}")
                        else:
                            raise
                    
                    # Store video-level metadata
                    video_group.attrs['video_label'] = label
                    video_group.attrs['is_fake'] = 1 if 'fake' in label.lower() else 0
                    video_group.attrs['original_filename'] = video_name
                    video_group.attrs['window_count'] = len(windows)
                    
                    for window_idx, signals in enumerate(windows):
                        if not signals:  # Skip empty signals
                            continue
                            
                        try:
                            window_group = video_group.create_group(f'window_{window_idx}')
                        except ValueError as e:
                            if "name already exists" in str(e).lower():
                                # Handle window name collision
                                alt_window_name = f'window_{window_idx}_{random.randint(100, 999)}'
                                window_group = video_group.create_group(alt_window_name)
                                logger.warning(f"Used alternative window name: {alt_window_name}")
                            else:
                                raise
                        
                        # Store window-level metadata
                        window_group.attrs['is_fake'] = 1 if 'fake' in label.lower() else 0
                        window_group.attrs['window_idx'] = window_idx
                        
                        # Save multi-ROI signals with proper structure
                        if 'multi_roi' in signals and signals['multi_roi']:
                            roi_group = window_group.create_group('multi_roi')
                            
                            # Store ROI metadata for shape inference
                            roi_names = list(signals['multi_roi'].keys())
                            roi_group.attrs['roi_names'] = roi_names
                            roi_group.attrs['roi_count'] = len(roi_names)
                            
                            for roi, roi_signals in signals['multi_roi'].items():
                                if not roi_signals:  # Skip empty ROI signals
                                    continue
                                    
                                roi_subgroup = roi_group.create_group(roi)
                                
                                # Store feature names for this ROI
                                feature_names = list(roi_signals.keys())
                                roi_subgroup.attrs['feature_names'] = feature_names
                                roi_subgroup.attrs['feature_count'] = len(feature_names)
                                
                                for k, v in roi_signals.items():
                                    try:
                                        # Ensure data is float32 and handle empty arrays
                                        data = np.array(v, dtype=np.float32)
                                        if data.size > 0:
                                            # Validate data before saving
                                            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                                                logger.warning(f"Invalid values in ROI {roi} feature {k}, cleaning...")
                                                data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                                            roi_subgroup.create_dataset(k, data=data, compression='gzip', compression_opts=6)
                                        else:
                                            # Create empty dataset with proper shape
                                            roi_subgroup.create_dataset(k, data=np.array([], dtype=np.float32))
                                    except Exception as e:
                                        logger.warning(f"Failed to save ROI signal {roi}/{k}: {e}")

                        # Save ML features
                        if 'ml_features' in signals and signals['ml_features']:
                            ml_group = window_group.create_group('ml_features')
                            
                            # Store metadata
                            feature_names = list(signals['ml_features'].keys())
                            ml_group.attrs['feature_names'] = feature_names
                            ml_group.attrs['feature_count'] = len(feature_names)
                            
                            # Save each feature as a scalar dataset
                            for feature_name, feature_value in signals['ml_features'].items():
                                try:
                                    # Ensure it's a scalar value and handle NaN/inf
                                    if np.isnan(feature_value) or np.isinf(feature_value):
                                        scalar_value = 0.0
                                        logger.warning(f"Invalid ML feature value for {feature_name}, using 0.0")
                                    else:
                                        scalar_value = float(feature_value)
                                    ml_group.create_dataset(feature_name, data=scalar_value)
                                except Exception as e:
                                    logger.warning(f"Failed to save ML feature {feature_name}: {e}")
                        
                        # Save face mask and window mask
                        for mask_name in ['face_mask', 'window_mask']:
                            if mask_name in signals:
                                try:
                                    mask_data = np.array(signals[mask_name], dtype=np.float32)
                                    if mask_data.size > 0:
                                        # Validate mask data
                                        if np.any(np.isnan(mask_data)) or np.any(np.isinf(mask_data)):
                                            logger.warning(f"Invalid values in {mask_name}, cleaning...")
                                            mask_data = np.nan_to_num(mask_data, nan=0.0, posinf=1.0, neginf=0.0)
                                        # Ensure mask values are in valid range [0, 1]
                                        mask_data = np.clip(mask_data, 0.0, 1.0)
                                        window_group.create_dataset(mask_name, data=mask_data, compression='gzip', compression_opts=6)
                                    else:
                                        window_group.create_dataset(mask_name, data=np.array([], dtype=np.float32))
                                except Exception as e:
                                    logger.warning(f"Failed to save {mask_name}: {e}")
                                    
    except Exception as e:
        logger.error(f"Failed to save HDF5 file {output_path}: {e}")
        # Clean up partially written file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Cleaned up corrupted file: {output_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up corrupted file {output_path}: {cleanup_error}")
        raise

def preprocess_dataset_for_dl(video_dir, output_dir, label, window_size=150, hop_size=75, max_workers=None, batch_size=50, augment_data=True):
    """
    OPTIMIZED: Process dataset with reduced feature extraction and improved resource management
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_file = os.path.join(output_dir, f"{label}_processed.txt")

    # Load previously processed videos
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            processed_videos = set(line.strip() for line in f)
    else:
        processed_videos = set()

    # Find videos to process
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    all_videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
    video_files = [f for f in all_videos if f not in processed_videos]

    logger.info(f"Found {len(video_files)} videos for {label}, {len(processed_videos)} already processed.")
    logger.info(f"OPTIMIZED: Using reduced feature set (20 ROI features per time step)")

    # Determine augmentation settings
    if augment_data:
        if label.lower() == 'real':
            should_augment = True
            augment_probability = 0.6
        elif label.lower() == 'fake':
            should_augment = True  
            augment_probability = 0.3
        else:
            should_augment = False
            augment_probability = 0.0
    else:
        should_augment = False
        augment_probability = 0.0
    
    # Batch processing setup
    batch_data = {}
    existing_batches = [f for f in os.listdir(output_dir) if re.match(rf"{label}_batch\d+_raw_signals\.h5", f)]
    if existing_batches:
        batch_indices = [int(re.search(rf"{label}_batch(\d+)_raw_signals\.h5", f).group(1)) for f in existing_batches]
        batch_idx = max(batch_indices) + 1
    else:
        batch_idx = 0
    n_in_batch = 0

    # Determine optimal worker count for AMD system
    if max_workers is None:
        # Conservative approach for AMD RX 6600 system
        max_workers = min(multiprocessing.cpu_count() // 2, 4)  # Limit to 4 workers max
    
    logger.info(f"Using {max_workers} workers for processing")
    
    args_list = [(video_file, video_dir, window_size, hop_size, should_augment) for video_file in video_files]

    # Use separate output files per process to avoid HDF5 conflicts
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs first
        future_to_args = {executor.submit(process_one_video, args): args for args in args_list}
        
        for future in tqdm(as_completed(future_to_args), total=len(future_to_args), desc=f"Processing {label} videos"):
            try:
                args = future_to_args[future]
                video_file = args[0]
                windows = future.result()
                
                if windows and len(windows) > 0:
                    # Save immediately to process-specific file
                    pid = os.getpid()
                    temp_output = os.path.join(output_dir, f'{label}_temp_{pid}_{batch_idx}_{n_in_batch}.h5')
                    save_raw_signals_hdf5(temp_output, {video_file: windows}, label)
                    
                    batch_data[video_file] = temp_output  # Store file path instead of data
                    n_in_batch += 1
                    logger.info(f"{video_file}: {len(windows)} windows extracted")
                    
                    # Mark as processed
                    with open(processed_file, "a") as pf:
                        pf.write(video_file + "\n")

                    # Merge temp files when batch is full
                    if n_in_batch >= batch_size:
                        output_path = os.path.join(output_dir, f'{label}_batch{batch_idx}_raw_signals.h5')
                        merge_temp_hdf5_files(list(batch_data.values()), output_path, label)
                        
                        # Clean up temp files
                        for temp_file in batch_data.values():
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                        
                        logger.info(f"Saved batch {batch_idx} with {n_in_batch} videos to {output_path}")
                        batch_data = {}
                        n_in_batch = 0
                        batch_idx += 1
                else:
                    logger.warning(f"No windows extracted from {video_file}")
                    
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                continue

        # Handle final batch
        if batch_data:
            output_path = os.path.join(output_dir, f'{label}_batch{batch_idx}_raw_signals.h5')
            merge_temp_hdf5_files(list(batch_data.values()), output_path, label)
            
            # Clean up temp files
            for temp_file in batch_data.values():
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            logger.info(f"Saved final batch {batch_idx} with {n_in_batch} videos to {output_path}")

# ========== FACE DETECTION AND TRACKING FUNCTIONS ==========

def get_face_net():
    """Initialize face detection network with fallback options"""
    # Try multiple model paths
    model_paths = [
        ('models/weights-prototxt.txt', 'models/res_ssd_300Dim.caffeModel'),
        ('weights-prototxt.txt', 'res_ssd_300Dim.caffeModel'),
        ('face_detection/weights-prototxt.txt', 'face_detection/res_ssd_300Dim.caffeModel')
    ]
    
    for face_proto, face_model in model_paths:
        if os.path.exists(face_proto) and os.path.exists(face_model):
            try:
                face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
                logger.info(f"Face detection model loaded from {face_proto}, {face_model}")
                return face_net
            except Exception as e:
                logger.warning(f"Failed to load face model from {face_proto}, {face_model}: {e}")
                continue
    
    # If no models found, try to use OpenCV's built-in cascade (fallback)
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            logger.warning("Using Haar cascade as fallback face detector")
            return cascade_path  # Return path for Haar cascade
    except:
        pass
    
    logger.error("No face detection model found. Please ensure face detection models are available.")
    return None

def detect_faces_dnn(frame, face_net, conf_threshold=0.5):
    """Detect faces using DNN or Haar cascade fallback"""
    try:
        if face_net is None or frame is None or frame.size == 0:
            return []
        
        h, w = frame.shape[:2]
        
        # Check if face_net is a string (Haar cascade path)
        if isinstance(face_net, str):
            # Use Haar cascade fallback
            try:
                cascade = cv2.CascadeClassifier(face_net)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 4)
                return [(x, y, w, h) for (x, y, w, h) in faces]
            except Exception as e:
                logger.warning(f"Haar cascade detection failed: {e}")
                return []
        
        # Use DNN detection
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
                # Ensure valid box dimensions
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
        
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return []

def iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes with validation"""
    # Add protection against invalid boxes
    if any(dim <= 0 for dim in [boxA[2], boxA[3], boxB[2], boxB[3]]):
        return 0.0
    
    # Ensure boxes are within reasonable bounds
    if any(coord < 0 for coord in boxA[:2]) or any(coord < 0 for coord in boxB[:2]):
        return 0.0
        
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    # Better division by zero protection
    union = boxAArea + boxBArea - interArea
    return interArea / max(union, 1e-10)

def robust_track_faces(all_boxes, max_lost=5, iou_threshold=0.3, max_distance=100):
    """
    Improved IoU and centroid-based tracker with better error handling
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
        
        # Validate boxes
        valid_boxes = []
        for box in boxes:
            if len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
                x, y, w, h = box
                if w > 0 and h > 0 and x >= 0 and y >= 0:
                    valid_boxes.append(box)
        
        if not valid_boxes:
            continue
        
        # Handle first frame or no active tracks
        if not active_tracks:
            for b in valid_boxes:
                tracks[face_id_counter] = [(frame_idx, b)]
                active_tracks[face_id_counter] = [frame_idx, b, 0]
                face_id_counter += 1
            continue
        
        # Get current track info
        track_ids = list(active_tracks.keys())
        track_boxes = [active_tracks[tid][1] for tid in track_ids]
        
        # Compute cost matrix using both IoU and centroid distance
        n_tracks = len(track_ids)
        n_boxes = len(valid_boxes)
        
        if n_tracks == 0 or n_boxes == 0:
            # Handle unmatched boxes
            for box in valid_boxes:
                tracks[face_id_counter] = [(frame_idx, box)]
                active_tracks[face_id_counter] = [frame_idx, box, 0]
                face_id_counter += 1
            continue
            
        cost_matrix = np.ones((n_tracks, n_boxes)) * 1000  # High cost for no match
        
        try:
            # Calculate centroids with error handling
            box_centroids = []
            track_centroids = []
            
            for box in valid_boxes:
                x, y, w, h = box
                box_centroids.append([x + w/2, y + h/2])
            
            for track_box in track_boxes:
                x, y, w, h = track_box
                track_centroids.append([x + w/2, y + h/2])
            
            box_centroids = np.array(box_centroids)
            track_centroids = np.array(track_centroids)
            
            # Compute distances
            if len(box_centroids) > 0 and len(track_centroids) > 0:
                distances = cdist(track_centroids, box_centroids)
                
                # Compute IoUs and combined costs
                for i, track_box in enumerate(track_boxes):
                    for j, box in enumerate(valid_boxes):
                        iou_score = iou(track_box, box)
                        distance = distances[i, j] if i < distances.shape[0] and j < distances.shape[1] else max_distance
                        
                        # Only consider assignment if IoU is above threshold OR distance is small
                        if iou_score > iou_threshold or distance < max_distance:
                            # Combined cost: weighted sum of (1-IoU) and normalized distance
                            iou_cost = 1 - iou_score
                            dist_cost = distance / max(max_distance, 1e-10)
                            cost_matrix[i, j] = 0.6 * iou_cost + 0.4 * dist_cost
                
                # Solve assignment problem using Hungarian algorithm
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                matched_boxes = set()
                
                # Process assignments
                for row, col in zip(row_indices, col_indices):
                    # Only accept assignment if cost is reasonable
                    if cost_matrix[row, col] < 0.9:  # Threshold for valid match
                        tid = track_ids[row]
                        box = valid_boxes[col]
                        
                        # Update track
                        tracks[tid].append((frame_idx, box))
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
                for j, box in enumerate(valid_boxes):
                    if j not in matched_boxes:
                        tracks[face_id_counter] = [(frame_idx, box)]
                        active_tracks[face_id_counter] = [frame_idx, box, 0]
                        face_id_counter += 1
                        
        except Exception as e:
            logger.warning(f"Tracking failed at frame {frame_idx}: {e}")
            # Fallback: create new tracks for all boxes
            for box in valid_boxes:
                tracks[face_id_counter] = [(frame_idx, box)]
                active_tracks[face_id_counter] = [frame_idx, box, 0]
                face_id_counter += 1
        
        # Clean up lost tracks
        for tid in list(active_tracks.keys()):
            if active_tracks[tid][2] > max_lost:
                del active_tracks[tid]
    
    return tracks

def extract_rgb_signal_track(frames, face_boxes):
    """Extract RGB signal from tracked faces with improved error handling"""
    rgb_signal = []
    face_mask = []
    
    for frame, box in zip(frames, face_boxes):
        if box is not None and frame is not None:
            try:
                # Add bounds checking
                x, y, w, h = box
                frame_h, frame_w = frame.shape[:2]
                
                # Ensure box is within frame bounds
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))
                w = max(1, min(w, frame_w - x))
                h = max(1, min(h, frame_h - y))
                
                # Ensure minimum ROI size
                if w < 10 or h < 10:
                    rgb_signal.append([0.0, 0.0, 0.0])
                    face_mask.append(0)
                    continue
                
                roi = frame[y:y+h, x:x+w]
                
                if roi.size > 0 and len(roi.shape) >= 3:
                    # Handle different channel formats
                    if roi.shape[2] >= 3:
                        # Split channels safely
                        try:
                            b, g, r = cv2.split(roi[:, :, :3])  # Only use first 3 channels
                            # Compute means with validation
                            r_mean = float(np.mean(r)) if r.size > 0 else 0.0
                            g_mean = float(np.mean(g)) if g.size > 0 else 0.0
                            b_mean = float(np.mean(b)) if b.size > 0 else 0.0
                            
                            rgb_signal.append([r_mean, g_mean, b_mean])
                            face_mask.append(1)
                        except Exception as e:
                            logger.warning(f"Error computing RGB means: {e}")
                            rgb_signal.append([0.0, 0.0, 0.0])
                            face_mask.append(0)
                    else:
                        rgb_signal.append([0.0, 0.0, 0.0])
                        face_mask.append(0)
                else:
                    rgb_signal.append([0.0, 0.0, 0.0])
                    face_mask.append(0)
                    
            except Exception as e:
                logger.warning(f"Error extracting RGB signal: {e}")
                rgb_signal.append([0.0, 0.0, 0.0])
                face_mask.append(0)
        else:
            rgb_signal.append([0.0, 0.0, 0.0])
            face_mask.append(0)
    
    return np.array(rgb_signal, dtype=np.float32), np.array(face_mask, dtype=np.float32)

# ============= MAIN EXECUTION ===================

if __name__ == "__main__":
    try:
        # Configuration for AMD RX 6600 system
        config = {
            'window_size': 150,
            'hop_size': 75,
            'max_workers': 10,  # Conservative for AMD system
            'batch_size': 50,  # Smaller batches for memory efficiency
            'augment_data': True
        }
        
        # Validate directories exist before processing
        real_video_dir = 'path/to/real/videos'
        fake_video_dir = 'path/to/fake/videos'
        output_dir = 'preprocessed_data_fixed/'
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Configuration: {config}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Process real videos
        if os.path.exists(real_video_dir):
            print("Processing real videos...")
            preprocess_dataset_for_dl(
                video_dir=real_video_dir,
                output_dir=output_dir,
                label='real',
                window_size=config['window_size'],
                hop_size=config['hop_size'],
                max_workers=config['max_workers'],
                batch_size=config['batch_size'],
                augment_data=config['augment_data']
            )
            print("Real videos processing completed.")
        else:
            print(f"Real video directory not found: {real_video_dir}")
            print("Please update the path to your real videos directory.")
        
        print()
        
        # Process fake videos  
        if os.path.exists(fake_video_dir):
            print("Processing fake videos...")
            preprocess_dataset_for_dl(
                video_dir=fake_video_dir,
                output_dir=output_dir,
                label='fake',
                window_size=config['window_size'],
                hop_size=config['hop_size'],
                max_workers=config['max_workers'],
                batch_size=config['batch_size'],
                augment_data=config['augment_data']
            )
            print("Fake videos processing completed.")
        else:
            print(f"Fake video directory not found: {fake_video_dir}")
            print("Please update the path to your fake videos directory.")
        
        print()
        print("=== PREPROCESSING COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise