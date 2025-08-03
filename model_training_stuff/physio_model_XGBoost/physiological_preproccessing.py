import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import detrend, butter, filtfilt, periodogram
import scipy.stats
import logging
import warnings
import random
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import mediapipe as mp
from antropy import sample_entropy
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ===== Seeding ======
# np.random.seed(43) # 42
# random.seed(43) # 42
 
MIN_FRAMES = 120
BASE_FEATURE_COUNT = 106
FINAL_FEATURE_COUNT = 110  # 106 + 4

# --- Augmentation Config ---
AUGMENTATION = {
    "enabled": True,
    # Probabilities
    "p_brightness": 0.25,
    "p_contrast": 0.25,
    "p_noise": 0.28,
    "p_color_shift": 0.25,
    "p_motion_blur": 0.2,
    "p_jpeg": 0.22,
    "p_gamma": 0.15,
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
        mean = np.nanmean(roi, axis=(0, 1), keepdims=True)
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
        
        color = np.nanmean(roi, axis=(0, 1)) + np.random.randint(-10, 11, size=(c,))
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

def choose_window_augmentations(aug_cfg):
    """Randomly select a set of augmentations and order for the whole window"""
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
    # Temporarily disabled cropping
    # if np.random.rand() < aug_cfg["p_crop"]:
    #     augmentations.append(lambda x: augment_random_crop(x, aug_cfg))
    np.random.shuffle(augmentations)
    
    return augmentations

def apply_augmentations(roi, augmentations):
    for func in augmentations:
        roi = func(roi)
    return roi

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
        logger.warning(f"CHROM method failed: {e}")
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
        logger.warning(f"Band power computation failed: {e}")
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
        logger.warning(f"Autocorrelation computation failed: {e}")
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
        logger.warning(f"Entropy computation failed: {e}")
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
            logger.warning("Filter produced invalid values, returning original signal")
            return signal
        # Put NaNs back in original positions
        filtered[nan_mask] = np.nan
        
        return filtered
    except Exception as e:
        logger.warning(f"Bandpass filter failed: {e}")
        return signal


def validate_rppg_features_simple(features, method_hr_bpm_values, method_snr_values):
    """
    NaN-aware sanity checks for rPPG features - only check clear garbage indicators
    
    Returns:
        is_valid (bool): Whether features pass basic sanity checks
        failure_reason (str): Single reason for failure (for debugging)
    """
    try:
        # 1. Basic feature validity
        if features is None or len(features) == 0:
            return False, "Empty features"
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False, "NaN/Inf in features"
        
        # 2. Check for NaN/Inf in features (this should not happen after proper processing)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            nan_count = np.sum(np.isnan(features))
            inf_count = np.sum(np.isinf(features))
            return False, f"NaN/Inf in features (NaN: {nan_count}, Inf: {inf_count})"
        
        # 3. At least one method should detect reasonable HR 
        hr_values = np.array(method_hr_bpm_values)
        # Remove NaN values before checking validity
        hr_values_clean = hr_values[~np.isnan(hr_values)]
        if len(hr_values_clean) == 0:
            return False, "All HR values are NaN"
        
        valid_hr_count = np.sum((hr_values_clean >= 30) & (hr_values_clean <= 210) & (hr_values_clean > 0))
        if valid_hr_count == 0:
            return False, f"No valid HR detected (range: {np.min(hr_values_clean):.1f}-{np.max(hr_values_clean):.1f})"
        
        # 4. At least one method should have decent SNR
        snr_values = np.array(method_snr_values)
        # Remove NaN values before checking validity
        snr_values_clean = snr_values[~np.isnan(snr_values)]
        if len(snr_values_clean) == 0:
            return False, "All SNR values are NaN"
        
        # 5. Features shouldn't be all zeros (dead signal)
        feature_abs_sum = np.nansum(np.abs(features))
        if feature_abs_sum < 1e-6:
            return False, "Zero features"
        
        # 6. Features should have some variance (not all identical)
        feature_std = np.nanstd(features)
        if feature_std < 1e-6:
            return False, "No feature variance"
        
        # 7. Check that dont have too many extreme outliers in features
        finite_features = features[np.isfinite(features)]
        if len(finite_features) < len(features) * 0.7:
            return False, f"Too many non-finite features ({len(finite_features)}/{len(features)})"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"
    
def compute_rppg_features_multi(rgb_signal, fs):
    """
    Compute rPPG features using multiple methods with simple sanity checks
    """
    try:
        if rgb_signal.size == 0 or rgb_signal.shape[0] < MIN_FRAMES:
            logger.warning("RGB signal too short or empty")
            return np.zeros(BASE_FEATURE_COUNT, dtype=np.float32), 0, np.zeros(rgb_signal.shape[0]), np.array([]), np.array([])
        
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
        
        # ===== SIMPLE SANITY CHECKS =====
        is_valid, failure_reason = validate_rppg_features_simple(
            features_array, method_hr_bpm_values, method_snr_values
        )
        
        if not is_valid:
            logger.warning(f"rPPG features rejected: {failure_reason}")
            # Return zeros for invalid features
            return np.zeros(BASE_FEATURE_COUNT, dtype=np.float32), 0, np.zeros(rgb_signal.shape[0]), np.array([]), np.array([])
        
        return (
            features_array, 
            mean_bpm/3, 
            rppg_methods["CHROM"], 
            f if 'f' in locals() else np.array([]), 
            pxx if 'pxx' in locals() else np.array([])
        )

    except Exception as e:
        logger.error(f"rPPG feature computation failed: {e}")
        return np.zeros(BASE_FEATURE_COUNT, dtype=np.float32), 0, np.zeros(rgb_signal.shape[0]), np.array([]), np.array([])

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

def detect_faces_dnn(frame, face_net, conf_threshold=0.6):
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
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                               borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded, scale, (left, top)

def create_facemesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

def get_skin_mask_mediapipe(frame, box, face_mesh):
    x, y, w, h = box
    x, y = max(x, 0), max(y, 0)
    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])
    face_roi = frame[y:y2, x:x2]
    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(face_roi_rgb)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
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
        return mask.astype(bool), True
    else:
        # Fallback to bounding box
        mask[y:y2, x:x2] = 1
        return mask.astype(bool), False


def extract_rgb_signal_track(frames, face_boxes, skin_masks):
    """Extract RGB signal from tracked faces using NaN-aware operations"""
    try:
        r_means, g_means, b_means, face_mask = [], [], [], []
        face_areas = []
        for idx, (frame, box) in enumerate(zip(frames, face_boxes)):
            mask = skin_masks[idx]
            if box is not None and frame is not None and frame.size > 0 and mask is not None:
                x, y, w, h = box
                x, y = max(x, 0), max(y, 0)
                x2 = min(x + w, frame.shape[1])
                y2 = min(y + h, frame.shape[0])
                if x2 > x and y2 > y:
                    roi = frame[y:y2, x:x2]
                    masked_pixels = mask[y:y2, x:x2]
                    num_skin_pixels = np.count_nonzero(masked_pixels)  # Count valid skin pixels
                    face_areas.append(num_skin_pixels)
                    if roi.size > 0 and len(roi.shape) == 3 and roi.shape[2] == 3 and np.any(masked_pixels):
                        r_mean = np.nanmean(roi[:,:,2][masked_pixels])
                        g_mean = np.nanmean(roi[:,:,1][masked_pixels])
                        b_mean = np.nanmean(roi[:,:,0][masked_pixels])
                        r_means.append(r_mean)
                        g_means.append(g_mean)
                        b_means.append(b_mean)
                        face_mask.append(1)
                    else:
                        r_means.append(np.nan)
                        g_means.append(np.nan)
                        b_means.append(np.nan)
                        face_mask.append(0)
                else:
                    r_means.append(np.nan)
                    g_means.append(np.nan)
                    b_means.append(np.nan)
                    face_mask.append(0)
            else:
                r_means.append(np.nan)
                g_means.append(np.nan)
                b_means.append(np.nan)
                face_mask.append(0)
        
        rgb_signal = np.stack([r_means, g_means, b_means], axis=-1)
        return rgb_signal, np.array(face_mask), np.array(face_areas)
    except Exception as e:
        logger.warning(f"RGB signal extraction failed: {e}")
        return np.full((len(frames), 3), np.nan), np.zeros(len(frames)), np.full((len(frames),), np.nan)

def sliding_windows_with_mask(signal, mask, window_size, hop_size, min_face_ratio=0.6):
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
    video_path, window_size, hop_size, aug_chance = args

    try:
        # Initialize face detection
        face_net = get_face_net()
        face_mesh = create_facemesh()
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
            frame, scale, (pad_left, pad_top) = letterbox_resize(frame, (640, 360))
            
            # Detect faces
            boxes = detect_faces_dnn(frame, face_net)
            
            frames.append(frame)
            all_boxes.append(boxes)
            frame_count += 1
        
        if cap is not None:
            cap.release()
        
        if len(frames) < MIN_FRAMES:
            logger.warning(f"Video {os.path.basename(video_path)} too short: {len(frames)} frames < {MIN_FRAMES}")
            return (None, 0, os.path.basename(video_path), 0, len(frames))

        # Track faces
        tracks = robust_track_faces(all_boxes)

        if not tracks or len(tracks) < 1:
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
                
                per_frame_skin_masks = [None] * len(frames)
                for idx, (frame, box) in enumerate(zip(frames, per_frame_boxes)):
                    if box is not None:
                        mask, _ = get_skin_mask_mediapipe(frame, box, face_mesh)
                        per_frame_skin_masks[idx] = mask
                    else:
                        per_frame_skin_masks[idx] = None
                
                # Extract RGB signal
                rgb_signal, face_mask, face_areas = extract_rgb_signal_track(frames, per_frame_boxes, per_frame_skin_masks)
                
                if len(frames) >= window_size:
                    # Standard behavior: use sliding windows
                    rgb_windows = sliding_windows_with_mask(rgb_signal, face_mask, window_size, hop_size, min_face_ratio=0.6)
                    window_lengths = [window_size] * len(rgb_windows)  # track the length for each window (all 150)
                else:
                    # For short videos, just make one window and pad
                    pad_len = window_size - len(frames)
                    rgb_padded = np.pad(rgb_signal, ((0, pad_len), (0,0)), constant_values=np.nan)
                    face_mask_padded = np.pad(face_mask, (0, pad_len), constant_values=0)
                    face_ratio = np.mean(face_mask_padded)  # Calculate valid face ratio

                    # Only keep this window if face ratio is above threshold
                    if face_ratio >= 0.6:
                        rgb_windows = np.expand_dims(rgb_padded, axis=0)
                        window_lengths = [len(frames)]
                    else:
                        rgb_windows = np.empty((0, window_size, 3))
                        window_lengths = []
                
                for win_idx, win in enumerate(rgb_windows):
                    if win.size > 0:
                        is_aug = AUGMENTATION and AUGMENTATION.get("enabled", False) and random.random() < aug_chance
                        if is_aug:
                            chosen_augs = choose_window_augmentations(AUGMENTATION)
                        else:
                            chosen_augs = []
                        # Get frame indices for this window
                        frame_indices = range(win_idx * hop_size, win_idx * hop_size + window_size)
                        window_face_areas = [face_areas[idx] if idx < len(face_areas) else 0 for idx in frame_indices]
                        avg_face_area = np.mean(window_face_areas)
                        med_face_area = np.median(window_face_areas)
                        # Only keep indices within bounds and where box exists
                        valid_idx = [i for i in frame_indices if i < len(frames) and per_frame_boxes[i] is not None]
                        window_rgb = []
                        for idx in frame_indices:
                            if idx < len(frames) and per_frame_boxes[idx] is not None:
                                x, y, w, h = per_frame_boxes[idx]
                                x, y = max(x, 0), max(y, 0)
                                x2 = min(x + w, frames[idx].shape[1])
                                y2 = min(y + h, frames[idx].shape[0])
                                if x2 > x and y2 > y:
                                    mask = per_frame_skin_masks[idx]
                                    roi = frames[idx][y:y2, x:x2]
                                    if is_aug and chosen_augs:
                                        roi = apply_augmentations(roi, chosen_augs)
                                    if mask is not None and roi.size > 0 and len(roi.shape) == 3 and roi.shape[2] == 3:
                                        masked_pixels = mask[y:y2, x:x2]
                                        if np.any(masked_pixels):
                                            r_mean = np.nanmean(roi[:,:,2][masked_pixels])
                                            g_mean = np.nanmean(roi[:,:,1][masked_pixels])
                                            b_mean = np.nanmean(roi[:,:,0][masked_pixels])
                                            window_rgb.append([r_mean, g_mean, b_mean])
                                        else:
                                            window_rgb.append([np.nan, np.nan, np.nan])
                                    else:
                                        window_rgb.append([np.nan, np.nan, np.nan])
                                else:
                                    window_rgb.append([np.nan, np.nan, np.nan])
                            else:
                                window_rgb.append([np.nan, np.nan, np.nan])

                        # Ensure correct window size
                        while len(window_rgb) < window_size:
                            window_rgb.append([np.nan, np.nan, np.nan])
                        win_arr = np.array(window_rgb, dtype=np.float32)
                        # Compute valid frame ratio
                        valid_ratio = np.sum(~np.isnan(win_arr[:,0])) / window_size
                        if valid_ratio < 0.8:
                            continue
                        # Add window_length as a feature
                        window_length_feat = window_lengths[win_idx] if win_idx < len(window_lengths) else window_size

                        rppg_feat, mean_bpm, rppg_sig_chrom, f, pxx = compute_rppg_features_multi(win_arr, fps)
                        if rppg_feat is not None and rppg_feat.size > 0:
                            # valid frame ratio as feature
                            rppg_feat = np.concatenate([rppg_feat, [valid_ratio]])
                            # Face size in pixels as feature
                            rppg_feat = np.concatenate([rppg_feat, [avg_face_area, med_face_area]])
                            # Window length as feature
                            rppg_feat = np.concatenate([rppg_feat, [window_length_feat]])
                            rppg_features_all_tracks.append(rppg_feat)
            except Exception as e:
                logger.warning(f"Track processing failed for track {track_id} in {os.path.basename(video_path)}: {e}")
                continue

        if rppg_features_all_tracks:
            rppg_features_all_tracks = np.stack(rppg_features_all_tracks, axis=0)
            logger.info(f"DEBUG - Processing successfully done for {os.path.basename(video_path)}")
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
    finally:
        if cap is not None:
            cap.release()
        gc.collect()

# ===== This the main function =====
def cache_batches_parallel(video_dir, label, class_idx, batch_size=128, cache_dir='cache/batches/train', window_size=150, hop_size=75, max_workers=None, start_batch_idx=0, aug_chance=0.5):
    """Process videos in parallel and cache batches with immediate saving and memory optimization"""
    try:
        # Validate inputs
        if not os.path.exists(video_dir):
            logger.error(f"Video directory does not exist: {video_dir}")
            return
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get video files
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

        all_success_log = os.path.join(cache_dir, f"success_{label}_all.txt")
        already_processed = set()
        if os.path.exists(all_success_log):
            with open(all_success_log, "r") as f:
                already_processed = set(line.strip() for line in f if line.strip())

        file_list = []
        for f in os.listdir(video_dir):
            if f.lower().endswith(video_extensions):
                if f not in already_processed:
                    video_path = os.path.join(video_dir, f)
                    if os.path.isfile(video_path):
                        file_list.append(video_path)
        random.shuffle(file_list)
        
        if not file_list:
            logger.error(f"No video files found in {video_dir}")
            return
        
        logger.info(f"Found {len(file_list)} video files for {label}")
        
        # Initialize batch storage with minimal memory footprint
        current_batch_features = []
        current_batch_labels = []
        current_batch_videos = []
        current_batch_size = 0
        batch_idx = start_batch_idx
        
        # Tracking variables
        processed_count = 0
        failed_count = 0
        all_success_videos = []
        failed_videos = []
        
        def save_current_batch():
            """Helper function to save current batch and reset buffers"""
            nonlocal current_batch_features, current_batch_labels, current_batch_videos
            nonlocal current_batch_size, batch_idx
            
            if current_batch_size == 0:
                return
                
            # Stack features and labels
            try:
                Xf = np.concatenate(current_batch_features, axis=0)
                Y = np.concatenate(current_batch_labels, axis=0)
                
                # Save batch files
                batch_path_x = os.path.join(cache_dir, f"{label}_Xrppg_batch_{batch_idx}.npy")
                batch_path_y = os.path.join(cache_dir, f"{label}_y_batch_{batch_idx}.npy")
                
                np.save(batch_path_x, Xf)
                np.save(batch_path_y, Y)
                logger.info(f"Saved batch {batch_idx}: {Xf.shape[0]} samples from {len(current_batch_videos)} videos")
                
                # Log successful videos for this batch
                log_file = os.path.join(cache_dir, f"success_{label}_batch_{batch_idx}.txt")
                with open(log_file, "w") as f:
                    for vidname in current_batch_videos:
                        f.write(f"{vidname}\n")
                
                batch_idx += 1
                
                # Clear memory immediately
                del Xf, Y
                
            except Exception as e:
                logger.error(f"Failed to save batch {batch_idx}: {e}")
            finally:
                # Reset batch buffers
                current_batch_features.clear()
                current_batch_labels.clear()
                current_batch_videos.clear()
                current_batch_size = 0
                gc.collect()  # Force garbage collection
        
        # Create worker tasks
        tasks = [(video_path, window_size, hop_size, aug_chance) for video_path in file_list]
        
        logger.info(f"Using 4 workers for {label} ({len(file_list)} videos)")

        # Process in chunks to manage memory better
        chunk_size = 500  # Process x num of videos at a time
        # Process videos in parallel with immediate batch saving         
        for chunk_start in range(0, len(tasks), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(tasks))
            chunk_tasks = tasks[chunk_start:chunk_end]
            
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(tasks)-1)//chunk_size + 1}")
            
            # Process chunk
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(
                    tqdm(
                        executor.map(preprocess_video_worker, chunk_tasks),
                        total=len(chunk_tasks),
                        desc=f"Processing {label} chunk"
                    )
                )
            
            # Process results immediately
            for feats, count, vname, _, n_frames in results:
                if feats is None or count == 0:
                    logger.warning(f"{vname}: No valid features extracted (frames: {n_frames})")
                    failed_videos.append(vname)
                    failed_count += 1
                    continue
                
                processed_count += 1
                all_success_videos.append(vname)
                
                # Split features into samples and add to current batch
                labels = np.full((feats.shape[0],), class_idx, dtype=np.int32)
                
                samples_to_add = feats.shape[0]
                start_idx = 0
                
                while start_idx < samples_to_add:
                    # Calculate how many samples we can add to current batch
                    space_left = batch_size - current_batch_size
                    samples_this_round = min(space_left, samples_to_add - start_idx)
                    end_idx = start_idx + samples_this_round
                    
                    # Add samples to current batch
                    current_batch_features.append(feats[start_idx:end_idx])
                    current_batch_labels.append(labels[start_idx:end_idx])
                    current_batch_videos.append(vname)  # Track which video contributed
                    current_batch_size += samples_this_round
                    
                    # Save batch if it's full
                    if current_batch_size >= batch_size:
                        save_current_batch()
                    
                    start_idx = end_idx
                
                # Clean up features immediately after processing
                del feats, labels
            
            # Clean up chunk results
            del results
            gc.collect()
        
        # Save any remaining samples as final batch
        if current_batch_size > 0:
            logger.info(f"Saving final partial batch with {current_batch_size} samples")
            save_current_batch()
        
        # Write comprehensive logs
        # All successful videos
        all_success_videos = sorted(list(set(all_success_videos)))
        all_success_log = os.path.join(cache_dir, f"success_{label}_all.txt")
        with open(all_success_log, "w") as f:
            for vidname in all_success_videos:
                f.write(f"{vidname}\n")
        
        # Failed videos
        if failed_videos:
            failed_log = os.path.join(cache_dir, f"failed_{label}_videos.txt")
            with open(failed_log, "w") as f:
                for vidname in failed_videos:
                    f.write(f"{vidname}\n")
        
        # Summary log
        summary_log = os.path.join(cache_dir, f"summary_{label}.txt")
        with open(summary_log, "w") as f:
            f.write(f"Total videos processed: {processed_count}\n")
            f.write(f"Total videos failed: {failed_count}\n")
            f.write(f"Total batches created: {batch_idx - start_batch_idx}\n")
            f.write(f"Batch size used: {batch_size}\n")
            f.write(f"Window size: {window_size}\n")
            f.write(f"Hop size: {hop_size}\n")
            f.write(f"Augmentation chance: {aug_chance}\n")
        
        logger.info(f"Completed {label}: {processed_count} videos processed, {failed_count} failed, {batch_idx - start_batch_idx} batches created")
        
        # Final cleanup
        current_batch_features.clear()
        current_batch_labels.clear()
        current_batch_videos.clear()
        gc.collect()
        
        return batch_idx

    except Exception as e:
        logger.error(f"Batch caching failed: {e}")
        # Ensure cleanup even on error
        try:
            current_batch_features.clear()
            current_batch_labels.clear()
            current_batch_videos.clear()
            gc.collect()
        except:
            pass
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
        # real_final_batch = cache_batches_parallel(
        #     video_dir="E:/deepfake_training_datasets/Physio_Model/TRAINING/real",
        #     label='real',
        #     class_idx=0,
        #     batch_size=128,
        #     cache_dir='C:/model_training/physio_ml/real',
        #     window_size=150,
        #     hop_size=75,
        #     start_batch_idx=287, # ========== ALWAYS REMEMBER TO UPDATE BATCH INDEX BEFORE EACH SUBSEQUENT RUNS ==========
        #     aug_chance=0.65
        # )

        # real_final_batch = cache_batches_parallel(
        #     video_dir="E:/deepfake_training_datasets/Physio_Model/TRAINING/real-semi-frontal",
        #     label='real',
        #     class_idx=0,
        #     batch_size=128,
        #     cache_dir='C:/model_training/physio_ml/real',
        #     window_size=150,
        #     hop_size=75,
        #     start_batch_idx=(real_final_batch + 1),
        #     aug_chance=0.6
        # )

        # fake_final_batch = cache_batches_parallel(
        #     video_dir="E:/deepfake_training_datasets/Physio_Model/TRAINING/fake",
        #     label='fake',
        #     class_idx=1,
        #     batch_size=128,
        #     cache_dir="C:/model_training/physio_ml/fake",
        #     window_size=150,
        #     hop_size=75,
        #     start_batch_idx=209, # ========== ALWAYS REMEMBER TO UPDATE BATCH INDEX BEFORE EACH SUBSEQUENT RUNS ==========
        #     aug_chance=0.425
        # )

        fake_final_batch = cache_batches_parallel(
            video_dir="E:/deepfake_training_datasets/Physio_Model/TRAINING/deeperforensics-fake-unedited",
            label='fake',
            class_idx=1,
            batch_size=128,
            cache_dir="C:/model_training/physio_ml/fake",
            window_size=150,
            hop_size=75,
            start_batch_idx=335,
            aug_chance=0.55
        )

        # real_final_batch = cache_batches_parallel(
        #     video_dir="F:/MP-Training-Datasets/real-celebvhq/35666",
        #     label='real',
        #     class_idx=0,
        #     batch_size=128,
        #     cache_dir='C:/model_training/physio_ml/real',
        #     window_size=150,
        #     hop_size=75,
        #     start_batch_idx=(real_final_batch + 1),
        #     aug_chance=0.4
        # )
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise