import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from scipy.signal import detrend, butter, filtfilt, hilbert
from scipy import linalg
import logging
import warnings
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import re

os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Global MediaPipe instance for reuse (fix #10)
_mp_face_mesh = None
_mp_face_mesh_solution = None

def get_mediapipe_instance():
    """Get reusable MediaPipe instance (fix #10)"""
    global _mp_face_mesh, _mp_face_mesh_solution
    if _mp_face_mesh is None:
        try:
            import mediapipe as mp
            _mp_face_mesh_solution = mp.solutions.face_mesh
            _mp_face_mesh = _mp_face_mesh_solution.FaceMesh(static_image_mode=True)
        except ImportError:
            logger.warning("MediaPipe not available, landmark detection will be skipped")
            return None, None
    return _mp_face_mesh, _mp_face_mesh_solution

# Cache for FFT computations (fix #11)
_fft_cache = {}

def cached_fft(signal, cache_key=None):
    """Compute FFT with caching to avoid redundant computations (fix #11)"""
    if cache_key is None:
        cache_key = hash(signal.tobytes())
    
    if cache_key not in _fft_cache:
        _fft_cache[cache_key] = np.fft.fft(signal)
        # Keep cache size reasonable
        if len(_fft_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(_fft_cache.keys())[:500]
            for k in keys_to_remove:
                del _fft_cache[k]
    
    return _fft_cache[cache_key]

# =================== HELPER FUNCTIONS ==========================
def get_face_landmarks(frame, box):
    """Use a facial landmark detector to get landmark points within a detected box."""
    face_mesh, _ = get_mediapipe_instance()
    if face_mesh is None:
        return None
        
    # Fix #3: Add bounds checking and handle different channel formats
    if len(frame.shape) != 3 or frame.shape[2] not in [3, 4]:
        logger.warning("Frame is not in expected BGR/RGB format")
        return None
        
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]
    
    # Bounds checking (fix #3)
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    
    roi_frame = frame[y:y+h, x:x+w]
    
    # Handle different channel formats (fix #3)
    if frame.shape[2] == 4:  # RGBA
        roi_frame = roi_frame[:, :, :3]  # Drop alpha channel
    
    rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
        
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Fix #4: Validate landmark indices and return coords relative to box
    valid_landmarks = []
    for l in landmarks:
        lx, ly = int(l.x * w), int(l.y * h)
        # Ensure landmarks are within bounds
        if 0 <= lx < w and 0 <= ly < h:
            valid_landmarks.append((lx, ly))
        else:
            valid_landmarks.append((0, 0))  # Fallback for invalid landmarks
            
    return valid_landmarks

def extract_multi_roi_signals(frame, box, landmarks, rois=['left_cheek', 'right_cheek', 'forehead', 'chin', 'nose']):
    """Extract mean RGB signals for each ROI using the landmarks."""
    # Fix #3: Add input validation
    if len(frame.shape) != 3 or frame.shape[2] < 3:
        logger.warning("Frame is not in expected BGR format")
        return {}
        
    if landmarks is None or len(landmarks) == 0:
        return {}
        
    h, w = box[3], box[2]
    
    # Fix #4: Use safer landmark indices with validation
    roi_points = {
        'left_cheek':  [234, 93, 132, 58],
        'right_cheek': [454, 323, 361, 288],
        'forehead':    [10, 338, 297, 332, 284, 251],
        'chin':        [152, 148, 176, 149, 150, 136],
        'nose':        [2, 98, 327, 195]
    }
    
    rgb_means = {}
    for roi in rois:
        # Fix #4: Validate landmark indices before accessing
        valid_indices = [i for i in roi_points[roi] if i < len(landmarks)]
        if len(valid_indices) < 3:  # Need at least 3 points for a polygon
            continue
            
        pts = np.array([landmarks[i] for i in valid_indices])
        mask = np.zeros((h, w), np.uint8)
        
        if pts.shape[0] > 2:
            cv2.fillPoly(mask, [pts], 1)
            
            # Fix #3: Add bounds checking for frame access
            x, y = box[0], box[1]
            frame_h, frame_w = frame.shape[:2]
            
            # Ensure we don't go out of bounds
            y_end = min(y + h, frame_h)
            x_end = min(x + w, frame_w)
            
            if y < frame_h and x < frame_w and y_end > y and x_end > x:
                roi_region = frame[y:y_end, x:x_end]
                
                # Adjust mask size if needed
                if roi_region.shape[:2] != mask.shape:
                    mask = mask[:roi_region.shape[0], :roi_region.shape[1]]
                
                # Mean RGB in ROI (fix #3: only process BGR channels)
                roi_means_list = []
                for c in range(min(3, roi_region.shape[2])):  # Only process first 3 channels
                    roi_pixels = roi_region[:, :, c][mask == 1]
                    roi_means_list.append(np.mean(roi_pixels) if roi_pixels.size > 0 else 0)
                
                # Pad to 3 channels if needed
                while len(roi_means_list) < 3:
                    roi_means_list.append(0)
                    
                rgb_means[roi] = roi_means_list[:3]  # Ensure exactly 3 values
    
    return rgb_means

def extract_signals_for_track(frames, per_frame_boxes, window_size, start, end, fps=30):
    """
    Extract both global and multi-ROI physiological signals for one track window.
    Returns:
        - raw_signals: dict (global face)
        - multi_roi_signals: dict of dicts (roi_name -> signals)
    """
    # Extract global RGB
    window_frames = frames[start:end]
    window_boxes = per_frame_boxes[start:end]

    # Global face signal
    rgb_signal, face_mask = extract_rgb_signal_track(window_frames, window_boxes)

    # Multi-ROI signals
    roi_signals = {}  # {roi: rgb_signal}
    roi_names = ['left_cheek', 'right_cheek', 'forehead', 'chin', 'nose']
    for roi in roi_names:
        roi_rgb_signal = []
        for frame, box in zip(window_frames, window_boxes):
            if box is not None:
                landmarks = get_face_landmarks(frame, box)
                if landmarks:
                    roi_means = extract_multi_roi_signals(frame, box, landmarks, rois=[roi])
                    vals = roi_means.get(roi, [0,0,0])
                    roi_rgb_signal.append(vals)
                else:
                    roi_rgb_signal.append([0,0,0])
            else:
                roi_rgb_signal.append([0,0,0])
        roi_signals[roi] = np.array(roi_rgb_signal)

    # Extract physio signals
    signals = {}
    # Global face signals
    signals['global'] = extract_raw_physio_signals(rgb_signal, fps)
    # Multi-ROI signals
    signals['multi_roi'] = {roi: extract_raw_physio_signals(roi_signals[roi], fps) for roi in roi_names}
    signals['face_mask'] = face_mask  # for QC
    
    return signals

def pad_and_mask(array, window_size, pad_mode='zero'):  # Fix #15: Default to zero padding
    """Pad array (1D or 2D) to window_size and return mask (1=real, 0=pad)."""
    array = np.array(array)
    L = array.shape[0]
    if L >= window_size:
        return array[:window_size], np.ones(window_size, dtype=np.float32)
    pad_len = window_size - L
    if array.ndim == 1:
        if L > 0:
            if pad_mode == 'repeat_last':
                pad = np.repeat(array[-1], pad_len)
            elif pad_mode == 'zero':
                pad = np.zeros(pad_len, dtype=array.dtype)
            else:
                pad = np.zeros(pad_len, dtype=array.dtype)  # Default to zero
            padded = np.concatenate([array, pad], axis=0)
        else:
            padded = np.zeros(window_size, dtype=array.dtype)
        mask = np.concatenate([np.ones(L, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)], axis=0)
    else:
        if L > 0:
            if pad_mode == 'repeat_last':
                pad = np.repeat(array[-1:], pad_len, axis=0)
            elif pad_mode == 'zero':
                pad = np.zeros((pad_len, array.shape[1]), dtype=array.dtype)
            else:
                pad = np.zeros((pad_len, array.shape[1]), dtype=array.dtype)  # Default to zero
            padded = np.concatenate([array, pad], axis=0)
        else:
            padded = np.zeros((window_size, array.shape[1]), dtype=array.dtype)
        mask = np.concatenate([np.ones(L, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)], axis=0)
    return padded, mask

# ========== ADVANCED rPPG EXTRACTION METHODS ==========

def rppg_chrom(rgb_signal):
    """CHROM method for rPPG extraction"""
    S = rgb_signal
    # Fix #5: Consistent minimum length handling
    min_length = 10
    if S.shape[0] < min_length:
        return np.zeros(S.shape[0])
    
    # Fix #12: Better division by zero protection
    S_std = np.std(S, axis=0)
    S_std = np.where(S_std < 1e-10, 1e-10, S_std)
    
    # Standardize
    S_norm = (S - np.mean(S, axis=0)) / S_std
    
    # CHROM combination
    h = S_norm[:, 1] - S_norm[:, 0]  # G - R
    s = S_norm[:, 1] + S_norm[:, 0] - 2 * S_norm[:, 2]  # G + R - 2B
    
    return h + s

def rppg_pos(rgb_signal):
    """POS (Plane Orthogonal to Skin) method"""
    S = rgb_signal
    # Fix #5: Consistent minimum length handling
    min_length = 10
    if S.shape[0] < min_length:
        return np.zeros(S.shape[0])
    
    # Spatial averaging
    S_mean = np.mean(S, axis=0)
    # Fix #12: Better division by zero protection
    S_mean = np.where(S_mean < 1e-10, 1e-10, S_mean)
    
    # Normalize
    S_norm = S / S_mean
    
    # POS combination
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Y = np.dot(S_norm, P.T)
    
    # Compute optimal projection
    if Y.shape[0] > 1:
        std_y = np.std(Y, axis=0)
        # Fix #12: Better division by zero protection
        std_y = np.where(std_y < 1e-10, 1e-10, std_y)
        Y_norm = Y / std_y
        
        # Alpha tuning with protection
        alpha_denom = np.std(Y_norm[:, 1])
        alpha = np.std(Y_norm[:, 0]) / max(alpha_denom, 1e-10)
        
        # Final signal
        rppg = Y_norm[:, 0] - alpha * Y_norm[:, 1]
    else:
        rppg = Y[:, 0]
    
    return rppg

def rppg_ica(rgb_signal):
    """ICA-based rPPG extraction"""
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        logger.warning("scikit-learn not available, falling back to CHROM")
        return rppg_chrom(rgb_signal)
    
    # Fix #5: Consistent minimum length handling
    min_length = 30  # ICA needs more samples
    if rgb_signal.shape[0] < min_length:
        return rppg_chrom(rgb_signal)  # Fallback
    
    try:
        # Center the signal
        rgb_centered = rgb_signal - np.mean(rgb_signal, axis=0)
        
        # Apply ICA
        ica = FastICA(n_components=3, random_state=42, max_iter=1000)
        sources = ica.fit_transform(rgb_centered)
        
        # Select component with highest power in HR frequency range
        best_source = None
        max_power = -1
        
        for i in range(sources.shape[1]):
            source = sources[:, i]
            # Simple frequency analysis using cached FFT (fix #11)
            cache_key = f"ica_{hash(source.tobytes())}"
            fft = cached_fft(source, cache_key)
            freqs = np.fft.fftfreq(len(source), d=1/30.0)  # Assume 30 fps
            
            # HR range: 0.7-4 Hz
            hr_mask = (np.abs(freqs) >= 0.7) & (np.abs(freqs) <= 4.0)
            hr_power = np.sum(np.abs(fft[hr_mask])**2)
            
            if hr_power > max_power:
                max_power = hr_power
                best_source = source
        
        return best_source if best_source is not None else sources[:, 0]
    except Exception as e:
        logger.warning(f"ICA failed: {e}, falling back to CHROM")
        return rppg_chrom(rgb_signal)  # Fallback to CHROM

def rppg_2sr(rgb_signal):
    """Two-Stage Regression (2SR) method"""
    # Fix #5: Consistent minimum length handling
    min_length = 30
    if rgb_signal.shape[0] < min_length:
        return rppg_chrom(rgb_signal)
    
    try:
        # Stage 1: Initial estimation using CHROM
        initial_pulse = rppg_chrom(rgb_signal)
        
        # Stage 2: Adaptive filtering
        # Estimate dominant frequency using cached FFT (fix #11)
        cache_key = f"2sr_{hash(initial_pulse.tobytes())}"
        fft = cached_fft(initial_pulse, cache_key)
        freqs = np.fft.fftfreq(len(initial_pulse), d=1/30.0)
        
        # Find peak in HR range
        hr_mask = (freqs >= 0.7) & (freqs <= 4.0)
        hr_fft = np.abs(fft[hr_mask])
        hr_freqs = freqs[hr_mask]
        
        if len(hr_fft) > 0:
            peak_idx = np.argmax(hr_fft)
            peak_freq = hr_freqs[peak_idx]
            
            # Adaptive bandpass filter around detected frequency
            lowcut = max(0.5, peak_freq - 0.5)
            highcut = min(4.5, peak_freq + 0.5)
            
            # Design adaptive filter
            try:
                b, a = butter(3, [lowcut, highcut], btype='band', fs=30.0)
                
                # Fix #8: Better filter stability check
                min_length_for_filter = max(len(a) * 6, 30)  # More conservative check
                
                # Apply to all channels and recombine
                filtered_rgb = np.zeros_like(rgb_signal)
                for i in range(3):
                    if len(rgb_signal[:, i]) >= min_length_for_filter:
                        filtered_rgb[:, i] = filtfilt(b, a, rgb_signal[:, i])
                    else:
                        filtered_rgb[:, i] = rgb_signal[:, i]
                
                # Refined estimation
                refined_pulse = rppg_pos(filtered_rgb)
                return refined_pulse
            except Exception as e:
                logger.warning(f"Filter design failed in 2SR: {e}")
                return initial_pulse
        else:
            return initial_pulse
    except Exception as e:
        logger.warning(f"2SR failed: {e}, falling back to CHROM")
        return rppg_chrom(rgb_signal)

def rppg_pbv(rgb_signal):
    """Pulse Blood Volume (PBV) method"""
    # Fix #5: Consistent minimum length handling
    min_length = 10
    if rgb_signal.shape[0] < min_length:
        return np.zeros(rgb_signal.shape[0])
    
    # Detrend
    rgb_detrend = detrend(rgb_signal, axis=0)
    
    # PBV signature extraction
    # Different combinations emphasize different aspects
    pbv_signatures = []
    
    # Classic combinations
    pbv_signatures.append(rgb_detrend[:, 1])  # Green channel
    pbv_signatures.append(rgb_detrend[:, 1] - rgb_detrend[:, 0])  # G - R
    pbv_signatures.append(2 * rgb_detrend[:, 1] - rgb_detrend[:, 0] - rgb_detrend[:, 2])  # 2G - R - B
    
    # Select best signature based on signal quality
    best_sig = pbv_signatures[0]
    max_quality = 0
    
    for sig in pbv_signatures:
        # Quality metric: ratio of power in HR band
        if len(sig) > 10:
            # Use cached FFT (fix #11)
            cache_key = f"pbv_{hash(sig.tobytes())}"
            fft = cached_fft(sig, cache_key)
            freqs = np.fft.fftfreq(len(sig), d=1/30.0)
            
            hr_mask = (np.abs(freqs) >= 0.7) & (np.abs(freqs) <= 4.0)
            total_power = np.sum(np.abs(fft)**2)
            hr_power = np.sum(np.abs(fft[hr_mask])**2)
            
            # Fix #12: Better division by zero protection
            quality = hr_power / max(total_power, 1e-10)
            if quality > max_quality:
                max_quality = quality
                best_sig = sig
    
    return best_sig

def rppg_omit(rgb_signal):
    """OMIT (Orthogonal Matrix Image Transform) method"""
    # Fix #5: Consistent minimum length handling
    min_length = 20
    if rgb_signal.shape[0] < min_length:
        return rppg_chrom(rgb_signal)
    
    try:
        # Remove mean
        rgb_centered = rgb_signal - np.mean(rgb_signal, axis=0)
        
        # Compute covariance matrix
        C = np.cov(rgb_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(C)
        
        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Project onto orthogonal subspace
        # The last eigenvector often contains pulse information
        pulse_direction = eigenvectors[:, -1]
        
        # Extract pulse signal
        rppg = np.dot(rgb_centered, pulse_direction)
        
        return rppg
    except Exception as e:
        logger.warning(f"OMIT failed: {e}, falling back to CHROM")
        return rppg_chrom(rgb_signal)

def rppg_lgi(rgb_signal):
    """Local Group Invariance (LGI) method"""
    # Fix #5: Consistent minimum length handling
    min_length = 20
    if rgb_signal.shape[0] < min_length:
        return rppg_chrom(rgb_signal)
    
    try:
        # Normalize channels with protection
        rgb_mean = np.mean(rgb_signal, axis=0)
        # Fix #12: Better division by zero protection
        rgb_mean = np.where(rgb_mean < 1e-10, 1e-10, rgb_mean)
        rgb_norm = rgb_signal / rgb_mean
        
        # Compute local derivatives
        rgb_diff = np.diff(rgb_norm, axis=0)
        
        # LGI transformation matrix (learned from data, here using empirical values)
        W = np.array([
            [0.5, -0.5, 0.0],    # R-G difference
            [0.25, 0.25, -0.5],  # (R+G)/2 - B
            [0.33, 0.33, 0.33]   # Intensity
        ])
        
        # Apply transformation
        if rgb_diff.shape[0] > 0:
            transformed = np.dot(rgb_diff, W.T)
            
            # Select pulse component (usually first or second)
            pulse_component = transformed[:, 0]
            
            # Integrate to get pulse signal
            rppg = np.cumsum(pulse_component)
            rppg = np.pad(rppg, (1, 0), mode='constant')  # Restore original length
        else:
            rppg = np.zeros(rgb_signal.shape[0])
        
        return rppg
    except Exception as e:
        logger.warning(f"LGI failed: {e}, falling back to CHROM")
        return rppg_chrom(rgb_signal)

def rppg_ssr(rgb_signal, fs=30):
    """Spatial Subspace Rotation (SSR) method"""
    # Fix #5: Consistent minimum length handling
    min_length = 30
    if rgb_signal.shape[0] < min_length:
        return rppg_chrom(rgb_signal)
    
    try:
        # Temporal normalization
        rgb_norm = detrend(rgb_signal, axis=0)
        
        # Build spatial correlation matrix
        R = np.dot(rgb_norm.T, rgb_norm) / rgb_norm.shape[0]
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        
        # Find pulse subspace (typically 2D)
        # Sort eigenvectors by eigenvalue
        idx = eigenvalues.argsort()
        U = eigenvectors[:, idx[:2]]  # Two smallest eigenvalues
        
        # Project signal onto pulse subspace
        S = np.dot(rgb_norm, U)
        
        # Estimate rotation angle using autocorrelation
        tau = int(fs * 0.5)  # 0.5 second lag
        if S.shape[0] > tau:
            C_tau = np.dot(S[:-tau].T, S[tau:]) / (S.shape[0] - tau)
            
            # Rotation angle with protection
            denom = C_tau[0, 0] - C_tau[1, 1]
            theta = 0.5 * np.arctan2(2 * C_tau[0, 1], denom if abs(denom) > 1e-10 else 1e-10)
            
            # Apply rotation
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            
            S_rotated = np.dot(S, rotation_matrix.T)
            rppg = S_rotated[:, 0]
        else:
            rppg = S[:, 0]
        
        return rppg
    except Exception as e:
        logger.warning(f"SSR failed: {e}, falling back to CHROM")
        return rppg_chrom(rgb_signal)

# ========== SIGNAL AUGMENTATION FOR DEEP LEARNING ==========

def augment_physiological_signal(signal, aug_type='none', strength=0.1, mask=None):
    """Apply physiologically plausible augmentations to rPPG signals"""
    if aug_type == 'none' or signal.size == 0:
        return signal
    
    augmented = signal.copy()
    
    # Fix #9: Apply augmentation only to real (non-padded) portions
    if mask is not None:
        real_indices = mask > 0.5  # Real data where mask is 1
        if not np.any(real_indices):
            return signal  # No real data to augment
    else:
        real_indices = np.ones(len(signal), dtype=bool)  # Augment all if no mask
    
    if aug_type == 'noise':
        # Add colored noise (more realistic than white noise)
        noise = np.random.randn(len(signal))
        # Apply 1/f filter to make pink noise
        if len(noise) > 1:
            fft = np.fft.fft(noise)
            freqs = np.fft.fftfreq(len(noise))
            nonzero_mask = freqs != 0
            if np.any(nonzero_mask):
                fft[nonzero_mask] = fft[nonzero_mask] / np.sqrt(np.abs(freqs[nonzero_mask]))
            noise = np.real(np.fft.ifft(fft))
            # Only apply to real portions
            augmented[real_indices] += strength * noise[real_indices] * np.std(signal[real_indices])
        
    elif aug_type == 'amplitude':
        # Random amplitude modulation - only on real portions
        real_length = np.sum(real_indices)
        if real_length > 0:
            modulation = 1 + strength * np.sin(2 * np.pi * np.random.rand() * np.arange(real_length) / real_length)
            augmented[real_indices] *= modulation
        
    elif aug_type == 'frequency':
        # Slight frequency shift (simulating HR variations) - only on real portions
        real_signal = signal[real_indices]
        if len(real_signal) > 1:
            shift_factor = 1 + np.random.uniform(-strength, strength)
            indices = np.arange(len(real_signal))
            new_indices = indices * shift_factor
            valid_mask = new_indices < len(real_signal)
            if np.any(valid_mask):
                interpolated = np.interp(indices, new_indices[valid_mask], 
                                       real_signal[:len(new_indices[valid_mask])])
                augmented[real_indices] = interpolated
        
    elif aug_type == 'phase':
        # Random phase shift - only on real portions
        real_signal = signal[real_indices]
        if len(real_signal) > 1:
            shift = int(strength * len(real_signal) * np.random.rand())
            augmented[real_indices] = np.roll(real_signal, shift)
        
    return augmented

# ========== PREPROCESSING FOR DEEP LEARNING ==========

def extract_raw_physio_signals(rgb_signal, fs=30):
    """
    Extract multiple raw physiological signals for deep learning
    Returns a dictionary of raw signals
    """
    signals = {}
    
    # Basic rPPG methods
    signals['chrom'] = rppg_chrom(rgb_signal)
    signals['pos'] = rppg_pos(rgb_signal)
    signals['ica'] = rppg_ica(rgb_signal)
    signals['2sr'] = rppg_2sr(rgb_signal)
    
    # Advanced methods
    signals['pbv'] = rppg_pbv(rgb_signal)
    signals['omit'] = rppg_omit(rgb_signal)
    signals['lgi'] = rppg_lgi(rgb_signal)
    signals['ssr'] = rppg_ssr(rgb_signal, fs)
    
    # Raw channels (useful for network to learn combinations)
    if rgb_signal.shape[1] >= 3:
        signals['green'] = rgb_signal[:, 1]
        signals['red'] = rgb_signal[:, 0]
        signals['blue'] = rgb_signal[:, 2]
        
        # Channel differences
        signals['g_r'] = rgb_signal[:, 1] - rgb_signal[:, 0]
        signals['g_b'] = rgb_signal[:, 1] - rgb_signal[:, 2]
        signals['r_b'] = rgb_signal[:, 0] - rgb_signal[:, 2]
    
    # Filtered versions (different frequency bands)
    for name, signal in list(signals.items()):
        if name in ['chrom', 'pos', 'ica', '2sr']:
            # Fix #8: Better filter stability check
            min_length_for_filter = max(30, 3 * 6)  # At least 6x filter order
            
            # Low frequency (0.5-2.5 Hz)
            try:
                if len(signal) >= min_length_for_filter:
                    b, a = butter(3, [0.5, 2.5], btype='band', fs=fs)
                    signals[f'{name}_low'] = filtfilt(b, a, signal)
            except Exception as e:
                logger.warning(f"Low-pass filtering failed for {name}: {e}")
            
            # High frequency (2.5-4.0 Hz)
            try:
                if len(signal) >= min_length_for_filter:
                    b, a = butter(3, [2.5, 4.0], btype='band', fs=fs)
                    signals[f'{name}_high'] = filtfilt(b, a, signal)
            except Exception as e:
                logger.warning(f"High-pass filtering failed for {name}: {e}")
    
    # Analytic signals (envelope)
    for name in ['chrom', 'pos']:
        try:
            if name in signals and len(signals[name]) > 10:
                analytic = hilbert(signals[name])
                signals[f'{name}_envelope'] = np.abs(analytic)
                signals[f'{name}_phase'] = np.angle(analytic)
        except Exception as e:
            logger.warning(f"Analytic signal computation failed for {name}: {e}")
    
    return signals

# ============= MAIN PREPROCESSING FUNCTIONS ==================
def process_one_video(args):
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
    Preprocess video for deep learning with raw signals
    Returns raw signal windows instead of features
    """
    try:
        # Fix #13: Validate video file exists before processing
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return None
        
        # Initialize face detection
        face_net = get_face_net()
        if face_net is None:
            logger.error(f"Face detection model not loaded")
            return None
        
        # Open video with better error handling (fix #13)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        frames = []
        all_boxes = []
        
        # Fix #13: Better FPS handling
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            logger.warning(f"Invalid FPS detected: {fps}, defaulting to 30")
            fps = 30.0
        
        # Read frames with better error handling (fix #13)
        frame_count = 0
        max_frames = 10000  # Prevent infinite loops
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Fix #13: Validate frame
            if frame is None or frame.size == 0:
                logger.warning(f"Empty frame at index {frame_count}")
                continue
            
            # Resize frame
            try:
                frame = cv2.resize(frame, (640, 360))
            except Exception as e:
                logger.warning(f"Frame resize failed at index {frame_count}: {e}")
                continue
            
            # Detect faces
            boxes = detect_faces_dnn(frame, face_net)
            
            frames.append(frame)
            all_boxes.append(boxes)
            frame_count += 1
        
        cap.release()
        
        if len(frames) < window_size:
            logger.warning(f"Video too short: {len(frames)} frames < {window_size}")
            return None
        
        # Track faces with better error handling (fix #7)
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

            # Sliding window
            n = len(frames)
            starts = list(range(0, max(n - window_size + 1, 1), hop_size))
            if n > 0 and (n - window_size) % hop_size != 0 and (n - window_size) > 0:
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

                # Fix #15: Use zero padding consistently
                padded_face_mask, mask_mask = pad_and_mask(signals['face_mask'], window_size, pad_mode='zero')

                # Pad all global and multi_roi signals with zero padding
                for key in signals['global']:
                    arr = np.array(signals['global'][key])
                    signals['global'][key], _ = pad_and_mask(arr, window_size, pad_mode='zero')
                for roi in signals['multi_roi']:
                    for key in signals['multi_roi'][roi]:
                        arr = np.array(signals['multi_roi'][roi][key])
                        signals['multi_roi'][roi][key], _ = pad_and_mask(arr, window_size, pad_mode='zero')
                
                # Fix #16: Use 60% threshold instead of hard-coded 90
                min_real_frames = int(0.6 * window_size)
                if np.sum(mask_mask) < min_real_frames:
                    continue  # Skip low-information (mostly-padded) window

                # Overwrite with padded face mask and add a 'mask' for real data positions
                signals['face_mask'] = padded_face_mask
                signals['window_mask'] = mask_mask  # Save for model

                # Only check stddev for the real, non-padded part!
                real_len = actual_end - start
                std_threshold = 1e-3
                stdval = np.std(signals['global']['chrom'][:real_len])
                if stdval < std_threshold:
                    continue

                roi_stds = [np.std(signals['multi_roi'][roi]['chrom'][:real_len]) for roi in signals['multi_roi']]
                if np.mean(roi_stds) < std_threshold:
                    continue

                # Fix #9: Improved augmentation with proper masking
                if augment:
                    aug_types = ['noise', 'amplitude', 'frequency', 'phase']
                    aug_type = np.random.choice(aug_types)
                    mask = signals['window_mask']  # (shape: [window_size], 1=real, 0=pad)
                    
                    # Apply augmentation to global signals
                    if 'global' in signals and np.random.rand() < 0.5:
                        for key in signals['global']:
                            signals['global'][key] = augment_physiological_signal(
                                signals['global'][key], 
                                aug_type=aug_type, 
                                strength=np.random.uniform(0.05, 0.15),
                                mask=mask
                            )
                    
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

                all_windows.append(signals)

        return all_windows
    except Exception as e:
        logger.error(f"Video preprocessing failed: {e}")
        return None

def sanitize_filename(filename):
    """Sanitize filename for HDF5 group names (fix #14)"""
    # Remove file extension and invalid characters
    name = os.path.splitext(filename)[0]
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    # Ensure name doesn't start with a number (HDF5 requirement)
    if name and name[0].isdigit():
        name = 'v_' + name
    # Ensure name is not empty
    if not name:
        name = 'unknown_video'
    return name

def save_raw_signals_hdf5(output_path, video_windows_dict, label):
    """
    Save both global and multi-ROI signals in HDF5 with proper labeling.
    FIXED: Add label parameter and store it properly
    """
    try:
        with h5py.File(output_path, 'w') as f:
            # CRITICAL FIX: Store dataset-level label
            f.attrs['dataset_label'] = label  # 'real', 'fake', etc.
            f.attrs['is_fake'] = 1 if 'fake' in label.lower() else 0
            
            for video_name, windows in video_windows_dict.items():
                # Fix #14: Sanitize video names for HDF5 compatibility
                safe_video_name = sanitize_filename(video_name)
                
                # Ensure unique group names
                counter = 1
                original_name = safe_video_name
                while safe_video_name in f:
                    safe_video_name = f"{original_name}_{counter}"
                    counter += 1
                
                video_group = f.create_group(safe_video_name)
                # CRITICAL FIX: Store video-level label for group-based splitting
                video_group.attrs['video_label'] = label
                video_group.attrs['is_fake'] = 1 if 'fake' in label.lower() else 0
                video_group.attrs['original_filename'] = video_name
                
                for window_idx, signals in enumerate(windows):
                    window_group = video_group.create_group(f'window_{window_idx}')
                    
                    # CRITICAL FIX: Store window-level label too
                    window_group.attrs['is_fake'] = 1 if 'fake' in label.lower() else 0
                    window_group.attrs['window_idx'] = window_idx
                    
                    # Save global signals
                    if 'global' in signals:
                        g = window_group.create_group('global')
                        for k, v in signals['global'].items():
                            try:
                                g.create_dataset(k, data=v, compression='gzip')
                            except Exception as e:
                                logger.warning(f"Failed to save global signal {k}: {e}")
                    
                    # CRITICAL FIX: Save multi-ROI signals with proper structure
                    if 'multi_roi' in signals:
                        roi_group = window_group.create_group('multi_roi')
                        
                        # Store ROI metadata for shape inference
                        roi_names = list(signals['multi_roi'].keys())
                        roi_group.attrs['roi_names'] = roi_names
                        
                        for roi, roi_signals in signals['multi_roi'].items():
                            roi_subgroup = roi_group.create_group(roi)
                            
                            # Store feature names for this ROI
                            feature_names = list(roi_signals.keys())
                            roi_subgroup.attrs['feature_names'] = feature_names
                            
                            for k, v in roi_signals.items():
                                try:
                                    roi_subgroup.create_dataset(k, data=v, compression='gzip')
                                except Exception as e:
                                    logger.warning(f"Failed to save ROI signal {roi}/{k}: {e}")
                    
                    # Save face mask and window mask
                    for mask_name in ['face_mask', 'window_mask']:
                        if mask_name in signals:
                            try:
                                window_group.create_dataset(mask_name, data=signals[mask_name], compression='gzip')
                            except Exception as e:
                                logger.warning(f"Failed to save {mask_name}: {e}")
                                
    except Exception as e:
        logger.error(f"Failed to save HDF5 file {output_path}: {e}")
        raise

def preprocess_dataset_for_dl(video_dir, output_dir, label, window_size=150, hop_size=75, max_workers=6, batch_size=100, augment_data=True):
    os.makedirs(output_dir, exist_ok=True)
    processed_file = os.path.join(output_dir, f"{label}_processed.txt")

    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            processed_videos = set(line.strip() for line in f)
    else:
        processed_videos = set()

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    all_videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
    video_files = [f for f in all_videos if f not in processed_videos]

    logger.info(f"Found {len(video_files)} videos for {label}, {len(processed_videos)} already processed.")

    batch_data = {}
    # Find existing batch files and set batch_idx accordingly
    existing_batches = [f for f in os.listdir(output_dir) if re.match(rf"{label}_batch\d+_raw_signals\.h5", f)]
    if existing_batches:
        batch_indices = [int(re.search(rf"{label}_batch(\d+)_raw_signals\.h5", f).group(1)) for f in existing_batches]
        batch_idx = max(batch_indices) + 1
    else:
        batch_idx = 0
    n_in_batch = 0

    if augment_data:
        if label.lower() == 'real':
            should_augment = True
            augment_probability = 0.6
        elif label.lower() == 'fake':
            should_augment = True  
            augment_probability = 0.3
        else:
            # Unknown label - be conservative
            should_augment = False
            augment_probability = 0.0
    else:
        should_augment = False
        augment_probability = 0.0
    
    args_list = [(video_file, video_dir, window_size, hop_size, should_augment, augment_probability) for video_file in video_files]

    with ProcessPoolExecutor(max_workers=max_workers or (multiprocessing.cpu_count() // 2)) as executor:
        futures = [executor.submit(process_one_video, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {label} videos"):
            video_file, windows = future.result()
            if windows:
                batch_data[video_file] = windows
                n_in_batch += 1
                logger.info(f"{video_file}: {len(windows)} windows extracted")
                with open(processed_file, "a") as pf:
                    pf.write(video_file + "\n")

                # Save the batch if full
                if n_in_batch >= batch_size:
                    output_path = os.path.join(output_dir, f'{label}_batch{batch_idx}_raw_signals.h5')
                    # CRITICAL FIX: Pass label to save function
                    save_raw_signals_hdf5(output_path, batch_data, label)
                    logger.info(f"Saved batch {batch_idx} with {n_in_batch} videos to {output_path}")
                    batch_data = {}
                    n_in_batch = 0
                    batch_idx += 1

        # Save the last partial batch (if any)
        if batch_data:
            output_path = os.path.join(output_dir, f'{label}_batch{batch_idx}_raw_signals.h5')
            # CRITICAL FIX: Pass label to save function
            save_raw_signals_hdf5(output_path, batch_data, label)
            logger.info(f"Saved final batch {batch_idx} with {n_in_batch} videos to {output_path}")

# ========== KEEP EXISTING FUNCTIONS FOR COMPATIBILITY ==========

def get_face_net():
    """Initialize face detection network"""
    try:
        FACE_PROTO = 'models/weights-prototxt.txt'
        FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'
        
        if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
            logger.error(f"Face detection model files not found")
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
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return []

def iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes."""
    # Fix #12: Add protection against invalid boxes
    if any(dim <= 0 for dim in [boxA[2], boxA[3], boxB[2], boxB[3]]):
        return 0.0
        
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    # Fix #12: Better division by zero protection
    union = boxAArea + boxBArea - interArea
    return interArea / max(union, 1e-10)

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
        
        # Fix #7: Handle edge cases for assignment
        if n_tracks == 0 or n_boxes == 0:
            # Handle unmatched boxes
            for box in boxes:
                tracks[face_id_counter] = [(frame_idx, box)]
                active_tracks[face_id_counter] = [frame_idx, box, 0]
                face_id_counter += 1
            continue
            
        cost_matrix = np.ones((n_tracks, n_boxes)) * 1000  # High cost for no match
        
        # Calculate centroids
        try:
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
                        dist_cost = distance / max(max_distance, 1e-10)  # Fix #12: protect division
                        cost_matrix[i, j] = 0.6 * iou_cost + 0.4 * dist_cost
            
            # Solve assignment problem using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            matched_boxes = set()
            
            # Process assignments
            for row, col in zip(row_indices, col_indices):
                # Only accept assignment if cost is reasonable
                if cost_matrix[row, col] < 0.9:  # Threshold for valid match
                    tid = track_ids[row]
                    box = boxes[col]
                    
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
            for j, box in enumerate(boxes):
                if j not in matched_boxes:
                    tracks[face_id_counter] = [(frame_idx, box)]
                    active_tracks[face_id_counter] = [frame_idx, box, 0]
                    face_id_counter += 1
                    
        except Exception as e:
            logger.warning(f"Tracking failed at frame {frame_idx}: {e}")
            # Fallback: create new tracks for all boxes
            for box in boxes:
                tracks[face_id_counter] = [(frame_idx, box)]
                active_tracks[face_id_counter] = [frame_idx, box, 0]
                face_id_counter += 1
        
        # Clean up lost tracks
        for tid in list(active_tracks.keys()):
            if active_tracks[tid][2] > max_lost:
                del active_tracks[tid]
    
    return tracks

def extract_rgb_signal_track(frames, face_boxes):
    """Extract RGB signal from tracked faces"""
    rgb_signal = []
    face_mask = []
    
    for frame, box in zip(frames, face_boxes):
        if box is not None:
            # Fix #3: Add bounds checking
            x, y, w, h = box
            frame_h, frame_w = frame.shape[:2]
            
            # Ensure box is within frame bounds
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            
            roi = frame[y:y+h, x:x+w]
            
            if roi.size > 0 and len(roi.shape) >= 3:
                # Fix #3: Handle different channel formats
                if roi.shape[2] >= 3:
                    b, g, r = cv2.split(roi[:, :, :3])  # Only use first 3 channels
                    rgb_signal.append([np.mean(r), np.mean(g), np.mean(b)])
                    face_mask.append(1)
                else:
                    rgb_signal.append([0, 0, 0])
                    face_mask.append(0)
            else:
                rgb_signal.append([0, 0, 0])
                face_mask.append(0)
        else:
            rgb_signal.append([0, 0, 0])
            face_mask.append(0)
    
    return np.array(rgb_signal), np.array(face_mask)

if __name__ == "__main__":
    try:
        # Process real videos
        preprocess_dataset_for_dl(
            video_dir='path/to/real/videos',
            output_dir='preprocessed_data/',
            label='real',  # CRITICAL: Use consistent labels
            window_size=150,
            hop_size=75,
            augment_data=True
        )
        
        # Process fake videos  
        preprocess_dataset_for_dl(
            video_dir='path/to/fake/videos',
            output_dir='preprocessed_data/',
            label='fake',  # CRITICAL: Use consistent labels
            window_size=150,
            hop_size=75,
            augment_data=True
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise