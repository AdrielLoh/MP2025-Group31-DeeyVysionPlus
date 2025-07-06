import os
import glob
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision
from sklearn.model_selection import StratifiedKFold, GroupKFold
import optuna
from tqdm import tqdm
import logging
import gc
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Setup for AMD DirectML ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def setup_amd_gpu():
    """Setup DirectML for AMD RX 6600 with memory optimization"""
    try:
        # Get available devices
        physical_devices = tf.config.list_physical_devices()
        logger.info(f"Available devices: {[d.name for d in physical_devices]}")
        
        # Check for DirectML
        dml_devices = tf.config.list_physical_devices('DML')
        if dml_devices:
            # Enable memory growth for DirectML
            for device in dml_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass  # Some DirectML versions don't support this
            logger.info(f"DirectML devices configured: {len(dml_devices)}")
        else:
            logger.warning("No DirectML devices found, using CPU")
        
        # Set memory limit if possible (for 8GB VRAM)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Limit GPU memory to 6GB to leave headroom
                tf.config.set_memory_growth(gpus[0], True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
                )
                logger.info("GPU memory limited to 6GB")
            except:
                logger.info("Could not set GPU memory limit, using dynamic growth")
        
        return True
    except Exception as e:
        logger.warning(f"DirectML setup failed: {e}")
        return False

def setup_mixed_precision_amd():
    """Setup mixed precision for AMD with DirectML compatibility"""
    try:
        # DirectML works better with mixed_float16
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled (float32)")
        return True
    except Exception as e:
        logger.warning(f"Mixed precision setup failed: {e}, using float32")
        return False

# --- Configuration ---
def get_config():
    """Get configuration optimized for AMD RX 6600"""
    parser = argparse.ArgumentParser(description='Train physiological deepfake detection model')
    parser.add_argument('--data_dir', type=str, default='preprocessed_data_fixed', 
                       help='Directory containing preprocessed HDF5 files')
    parser.add_argument('--model_dir', type=str, default='outputs/models',
                       help='Directory to save trained models')
    parser.add_argument('--log_dir', type=str, default='outputs/logs',
                       help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--optuna_trials', type=int, default=15,
                       help='Number of Optuna trials for hyperparameter search')
    parser.add_argument('--optuna_timeout', type=int, default=7200,  # 2 hours
                       help='Optuna timeout in seconds')
    parser.add_argument('--window_size', type=int, default=150,
                       help='Expected window size')
    parser.add_argument('--skip_optuna', action='store_true',
                       help='Skip hyperparameter optimization')
    parser.add_argument('--batch_size', type=int, default=8,  # Smaller for AMD
                       help='Batch size (optimized for 8GB VRAM)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    return args

# --- Updated Data Loader for New Feature Structure ---

def extract_label_from_filename(filename):
    """Extract label from filename with fallback to HDF5 attributes"""
    filename_path = filename
    filename = os.path.basename(filename).lower()
    
    # Primary method: check filename patterns
    if 'fake' in filename or 'deepfake' in filename or 'synthetic' in filename:
        return 1
    elif 'real' in filename or 'authentic' in filename or 'genuine' in filename:
        return 0
    else:
        # Fallback to HDF5 attributes
        try:
            with h5py.File(filename_path, 'r') as f:
                if 'dataset_label' in f.attrs:
                    label_str = f.attrs['dataset_label']
                    if isinstance(label_str, (bytes, np.bytes_)):
                        label_str = label_str.decode('utf-8')
                    return 1 if 'fake' in str(label_str).lower() else 0
                elif 'is_fake' in f.attrs:
                    return int(f.attrs['is_fake'])
        except Exception as e:
            logger.warning(f"Could not read label from {filename_path}: {e}")
        
        logger.warning(f"Could not determine label for {filename}, assuming real (0)")
        return 0

def extract_video_id(filename):
    """Extract video ID for group-based splitting"""
    try:
        with h5py.File(filename, 'r') as f:
            video_ids = []
            for vid_name in f.keys():
                vid_group = f[vid_name]
                if 'original_filename' in vid_group.attrs:
                    orig_name = vid_group.attrs['original_filename']
                    if isinstance(orig_name, (bytes, np.bytes_)):
                        orig_name = orig_name.decode('utf-8')
                    video_id = os.path.splitext(orig_name)[0]
                    video_ids.append(video_id)
                else:
                    video_ids.append(vid_name)
            return '_'.join(sorted(set(video_ids)))
    except Exception as e:
        logger.warning(f"Could not extract video ID from HDF5 {filename}: {e}")
    
    basename = os.path.basename(filename)
    return basename.replace('.h5', '')

def validate_hdf5_file(filepath):
    """Validate HDF5 file structure for new format"""
    try:
        with h5py.File(filepath, 'r') as f:
            if len(f.keys()) == 0:
                return False
            
            for vid_name in f.keys():
                vid_group = f[vid_name]
                if len(vid_group.keys()) == 0:
                    continue
                    
                for win_name in vid_group.keys():
                    win_group = vid_group[win_name]
                    
                    # Check required datasets
                    required_datasets = ['face_mask', 'window_mask']
                    for dataset in required_datasets:
                        if dataset not in win_group:
                            return False
                    
                    # Check multi_roi structure (NEW FORMAT)
                    if 'multi_roi' not in win_group:
                        return False
                    
                    roi_group = win_group['multi_roi']
                    expected_rois = ['left_cheek', 'right_cheek', 'forehead', 'chin', 'nose']
                    
                    # Check that we have the expected ROIs
                    available_rois = list(roi_group.keys())
                    if len(available_rois) == 0:
                        return False
                    
                    # Check that each ROI has the expected features
                    for roi_name in available_rois:
                        roi_subgroup = roi_group[roi_name]
                        expected_features = ['chrom', 'pos', 'ica']
                        
                        for feature in expected_features:
                            if feature not in roi_subgroup:
                                logger.warning(f"Missing feature {feature} in ROI {roi_name}")
                                return False
                    
                    return True  # Found valid structure
        return False
    except Exception as e:
        logger.warning(f"File validation failed for {filepath}: {e}")
        return False

def load_window_new_format(h5_window, expected_window_size=150):
    """Load window data in new format: multi-ROI only with 3 features per ROI"""
    try:
        # Load multi-ROI features (NEW FORMAT)
        if 'multi_roi' not in h5_window:
            raise ValueError("multi_roi group not found")
            
        roi_group = h5_window['multi_roi']
        available_rois = sorted(roi_group.keys())  # Consistent ordering
        
        # Expected ROI features: 3 per ROI (chrom, pos, ica)
        all_roi_features = []
        
        for roi in available_rois:
            r_dict = roi_group[roi]
            roi_data = []
            
            # Extract features in consistent order
            for feature_name in ['chrom', 'pos', 'ica']:
                if feature_name in r_dict:
                    data = r_dict[feature_name][()]
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        logger.warning(f"Invalid values in ROI {roi} feature {feature_name}")
                        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                    roi_data.append(data)
                else:
                    # Missing feature - use zeros
                    logger.warning(f"Missing feature {feature_name} in ROI {roi}")
                    roi_data.append(np.zeros(expected_window_size, dtype=np.float32))
            
            if roi_data:
                # Stack features for this ROI: (time, 3)
                roi_features = np.stack(roi_data, axis=-1)
                all_roi_features.append(roi_features)
        
        if not all_roi_features:
            raise ValueError("No ROI features found")
        
        # Concatenate all ROI features: (time, 5*3=15)
        X_roi_combined = np.concatenate(all_roi_features, axis=-1)
        
        # Load masks
        window_mask = h5_window['window_mask'][()]
        
        # Validate dimensions
        if X_roi_combined.shape[0] != expected_window_size:
            logger.warning(f"ROI features shape mismatch: {X_roi_combined.shape[0]} != {expected_window_size}")
        if window_mask.shape[0] != expected_window_size:
            logger.warning(f"Window mask shape mismatch: {window_mask.shape[0]} != {expected_window_size}")
        
        return (X_roi_combined.astype(np.float32),
                window_mask.astype(np.float32))
        
    except Exception as e:
        logger.error(f"Failed to load window: {e}")
        raise

def count_samples_and_get_shapes(file_paths):
    """Count total samples and get feature shapes for new format"""
    total_samples = 0
    n_roi_features = None
    labels = []
    
    for filepath in tqdm(file_paths, desc="Counting samples"):
        if not validate_hdf5_file(filepath):
            logger.warning(f"Skipping invalid file: {filepath}")
            continue
            
        file_label = extract_label_from_filename(filepath)
        
        try:
            with h5py.File(filepath, 'r') as f:
                for vid_name in f.keys():
                    vid_group = f[vid_name]
                    for win_name in vid_group.keys():
                        win_group = vid_group[win_name]
                        
                        # Get shapes from first valid window
                        if n_roi_features is None:
                            roi_dict = win_group['multi_roi']
                            n_roi_features = 0
                            
                            for roi_name in roi_dict.keys():
                                roi_subgroup = roi_dict[roi_name]
                                # Count features in this ROI (should be 3: chrom, pos, ica)
                                n_roi_features += len([k for k in roi_subgroup.keys() 
                                                     if k in ['chrom', 'pos', 'ica']])
                        
                        total_samples += 1
                        labels.append(file_label)
        except Exception as e:
            logger.warning(f"Error reading {filepath}: {e}")
            continue
    
    # For new format: 5 ROIs * 3 features = 15 total features
    if n_roi_features is None:
        n_roi_features = 15  # Default expected
    
    logger.info(f"Found {total_samples} samples, {n_roi_features} ROI features total")
    return total_samples, n_roi_features, np.array(labels)

def create_data_generator_new_format(file_paths, batch_size, n_roi_features, window_size=150, shuffle=True, infinite=True):
    """Create efficient data generator for new format"""
    def generator():
        file_indices = list(range(len(file_paths)))
        
        while True:
            if shuffle:
                np.random.shuffle(file_indices)
            
            batch_roi = []
            batch_mask = []
            batch_labels = []
            
            for file_idx in file_indices:
                filepath = file_paths[file_idx]
                file_label = extract_label_from_filename(filepath)
                
                try:
                    with h5py.File(filepath, 'r') as f:
                        for vid_name in f.keys():
                            vid_group = f[vid_name]
                            for win_name in vid_group.keys():
                                win_group = vid_group[win_name]
                                
                                try:
                                    X_roi_combined, window_mask = load_window_new_format(win_group, window_size)
                                    
                                    batch_roi.append(X_roi_combined)
                                    batch_mask.append(window_mask)
                                    batch_labels.append(file_label)
                                    
                                    if len(batch_roi) == batch_size:
                                        yield (np.stack(batch_roi),
                                               np.stack(batch_mask),
                                               np.array(batch_labels))
                                        
                                        batch_roi = []
                                        batch_mask = []
                                        batch_labels = []
                                        
                                except Exception as e:
                                    logger.warning(f"Skipping window {vid_name}/{win_name}: {e}")
                                    continue
                                    
                except Exception as e:
                    logger.warning(f"Skipping file {filepath}: {e}")
                    continue
            
            # Yield remaining batch if not empty
            if batch_roi and len(batch_roi) > 0:
                yield (np.stack(batch_roi),
                       np.stack(batch_mask),
                       np.array(batch_labels))
            
            if not infinite:
                break
    
    return generator

def make_dataset_new_format(file_paths, batch_size, n_roi_features, window_size=150, shuffle=True):
    """Create TensorFlow dataset for new format"""
    generator_fn = create_data_generator_new_format(file_paths, batch_size, n_roi_features, window_size, shuffle, infinite=False)
    
    output_signature = (
        tf.TensorSpec(shape=(None, window_size, n_roi_features), dtype=tf.float32),  # ROI features only
        tf.TensorSpec(shape=(None, window_size), dtype=tf.float32),  # Window mask
        tf.TensorSpec(shape=(None,), dtype=tf.int32)  # Labels
    )
    
    ds = tf.data.Dataset.from_generator(generator_fn, output_signature=output_signature)
    
    # Optimized for AMD RX 6600
    if shuffle:
        buffer_size = min(max(batch_size * 8, 64), 512)  # Smaller buffer for memory
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)
    
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- Updated Model Architecture ---

def build_tcn_transformer(cfg, use_mixed_precision=False):
    window_size = cfg.get('window_size', 150)
    n_roi_features = cfg['n_roi_features']
    tcn_channels = cfg.get('filters', 48)
    ff_dim = cfg.get('dense_dim', 96)
    num_heads = 4  # you can tune this too
    attn_dropout = cfg.get('dropout', 0.2)
    tcn_blocks = [2 ** i for i in range(cfg['blocks'])]

    # Input layers
    roi_in = layers.Input((window_size, n_roi_features), name='roi_in')
    mask = layers.Input((window_size,), name='mask_in')

    # TCN backbone
    x = roi_in
    for d in tcn_blocks:
        res = x
        x = layers.Conv1D(tcn_channels, 3, padding="causal", dilation_rate=d, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(tcn_channels, 3, padding="causal", dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([res, x])
        x = layers.Activation("relu")(x)

    # Downsample and project
    x = layers.MaxPooling1D(2)(x)  # shape: (window_size//2, tcn_channels)
    x = layers.Conv1D(ff_dim, 1, activation="relu")(x)  # shape: (window_size//2, ff_dim)

    # Positional embeddings
    seq_len = tf.shape(x)[1]
    positions = tf.range(seq_len)
    pos_emb = layers.Embedding(input_dim=window_size, output_dim=ff_dim)(positions)
    x = x + pos_emb

    # Single Transformer block
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=ff_dim // num_heads,
        dropout=attn_dropout
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(ff_dim)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    # Masked pooling (to handle window_mask, as in your old model)
    def masked_gap(args):
        f, m = args
        m = tf.cast(tf.expand_dims(m, -1), f.dtype)
        return tf.reduce_sum(f * m, 1) / tf.clip_by_value(tf.reduce_sum(m, 1, keepdims=True), 1e-3, tf.float32.max)
    pooled = layers.Lambda(masked_gap, name='masked_pooling')([x, mask[:, :tf.shape(x)[1]]])

    # Classification head
    dense = layers.Dense(ff_dim, name='dense_1')(pooled)
    dense = layers.ReLU()(dense)
    dense = layers.Dropout(attn_dropout)(dense)
    dense = layers.Dense(ff_dim // 2, name='dense_2')(dense)
    dense = layers.ReLU()(dense)
    dense = layers.Dropout(attn_dropout)(dense)
    if use_mixed_precision:
        out = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(dense)
    else:
        out = layers.Dense(1, activation='sigmoid', name='output')(dense)
    model = models.Model([roi_in, mask], out, name='TCN_Transformer')
    return model


# --- Training utilities ---

def compute_class_weights_efficient(file_paths):
    """Efficiently compute class weights without loading all data"""
    n_real = 0
    n_fake = 0
    
    for filepath in file_paths:
        label = extract_label_from_filename(filepath)
        
        try:
            with h5py.File(filepath, 'r') as f:
                file_window_count = sum(len(vid_group.keys()) for vid_group in f.values())
                
                if label == 0:
                    n_real += file_window_count
                else:
                    n_fake += file_window_count
        except Exception as e:
            logger.warning(f"Error counting windows in {filepath}: {e}")
            continue
    
    total = n_real + n_fake
    if total == 0:
        return {0: 1.0, 1: 1.0}
    
    w_real = total / (2.0 * n_real) if n_real > 0 else 1.0
    w_fake = total / (2.0 * n_fake) if n_fake > 0 else 1.0
    
    logger.info(f"Class weights: Real={w_real:.3f} (n={n_real}), Fake={w_fake:.3f} (n={n_fake})")
    return {0: w_real, 1: w_fake}

def evaluate_model_efficiently(model, val_dataset):
    """Efficiently evaluate model and collect predictions"""
    y_true = []
    y_pred = []
    
    for batch in val_dataset:
        Xr, Xm, yb = batch
        preds = model([Xr, Xm], training=False)
        y_true.extend(yb.numpy())
        y_pred.extend(preds.numpy().squeeze())
    
    return np.array(y_true), np.array(y_pred)

# --- Optuna objective ---

def objective(trial, train_files, val_files, n_roi_features, window_size, use_mixed_precision):
    """Optuna objective optimized for AMD"""
    try:
        cfg = {
            'blocks': trial.suggest_int('blocks', 3, 6),
            'filters': trial.suggest_categorical('filters', [32, 48, 64]),  # Smaller for AMD
            'dense_dim': trial.suggest_categorical('dense_dim', [64, 96, 128]),
            'lr': trial.suggest_float('lr', 5e-5, 2e-3, log=True),
            'batch': trial.suggest_categorical('batch', [4, 6, 8]),  # Small batches for 8GB VRAM
            'dropout': trial.suggest_float('dropout', 0.1, 0.3),
            'n_roi_features': n_roi_features,
            'window_size': window_size
        }
        
        # Create datasets
        train_ds = make_dataset_new_format(train_files, cfg['batch'], n_roi_features, window_size, shuffle=True)
        val_ds = make_dataset_new_format(val_files, cfg['batch'], n_roi_features, window_size, shuffle=False)
        
        # Build model
        model = build_tcn_transformer(cfg, use_mixed_precision)
        
        # Compile with optimizer suitable for DirectML
        optimizer = optimizers.Adam(cfg['lr'], clipnorm=1.0)  # Gradient clipping for stability
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[tf.keras.metrics.AUC(name='auroc')]
        )
        
        # Train with early stopping
        es = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_auroc', mode='max')
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=8,  # Shorter for hyperparameter search
            verbose=0,
            callbacks=[es]
        )
        
        # Get best validation score
        best_auroc = max(history.history['val_auroc'])
        
        # Clean up
        del model
        del train_ds
        del val_ds
        tf.keras.backend.clear_session()
        gc.collect()
        
        return 1.0 - best_auroc  # Optuna minimizes
        
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        tf.keras.backend.clear_session()
        gc.collect()
        return 1.0

# --- Main training function ---

def main():
    # Setup
    config = get_config()
    amd_setup_success = setup_amd_gpu()
    use_mixed_precision = setup_mixed_precision_amd() and amd_setup_success
    
    logger.info(f"Starting training with config: {vars(config)}")
    logger.info(f"AMD DirectML setup: {amd_setup_success}")
    logger.info(f"Mixed precision: {use_mixed_precision}")
    
    # Find and validate files
    if not os.path.exists(config.data_dir):
        logger.error(f"Data directory does not exist: {config.data_dir}")
        return
    
    all_files = sorted(glob.glob(os.path.join(config.data_dir, "*.h5")))
    if not all_files:
        logger.error(f"No HDF5 files found in {config.data_dir}")
        return
    
    logger.info(f"Found {len(all_files)} HDF5 files")
    
    # Validate files and get dataset info
    valid_files = []
    for filepath in tqdm(all_files, desc="Validating files"):
        if validate_hdf5_file(filepath):
            valid_files.append(filepath)
        else:
            logger.warning(f"Skipping invalid file: {filepath}")
    
    if not valid_files:
        logger.error("No valid files found")
        return
    
    logger.info(f"Using {len(valid_files)} valid files")
    
    # Get dataset information (NEW FORMAT)
    total_samples, n_roi_features, file_labels = count_samples_and_get_shapes(valid_files)
    logger.info(f"Dataset: {total_samples} samples, {n_roi_features} ROI features")
    
    # Extract video IDs for group-based splitting
    video_ids = [extract_video_id(f) for f in valid_files]
    
    # Use GroupKFold to prevent data leakage
    if len(set(video_ids)) >= config.folds:
        kfold = GroupKFold(n_splits=config.folds)
        splits = list(kfold.split(valid_files, file_labels, video_ids))
    else:
        logger.warning("Not enough unique videos for GroupKFold, using StratifiedKFold")
        kfold = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=42)
        splits = list(kfold.split(valid_files, file_labels))
    
    # Training loop
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n===== Fold {fold+1}/{config.folds} =====")
        
        train_files = [valid_files[i] for i in train_idx]
        val_files = [valid_files[i] for i in val_idx]
        
        logger.info(f"Train: {len(train_files)} files | Val: {len(val_files)} files")
        
        # Hyperparameter optimization
        if not config.skip_optuna:
            logger.info("Starting hyperparameter optimization...")
            
            def fold_objective(trial):
                return objective(trial, train_files, val_files, n_roi_features, config.window_size, use_mixed_precision)
            
            study = optuna.create_study(direction='minimize')
            study.optimize(fold_objective, n_trials=config.optuna_trials, timeout=config.optuna_timeout)
            
            best_cfg = {
                **study.best_params,
                'n_roi_features': n_roi_features,
                'window_size': config.window_size
            }
            logger.info(f"Best config for fold {fold+1}: {best_cfg}")
        else:
            # Use default configuration optimized for AMD
            best_cfg = {
                'blocks': 4,
                'filters': 48,
                'dense_dim': 96,
                'lr': 8e-4,
                'batch': config.batch_size,
                'dropout': 0.2,
                'n_roi_features': n_roi_features,
                'window_size': config.window_size
            }
            logger.info(f"Using default config for fold {fold+1}: {best_cfg}")
        
        # Create datasets
        train_ds = make_dataset_new_format(train_files, best_cfg['batch'], n_roi_features, config.window_size, shuffle=True)
        val_ds = make_dataset_new_format(val_files, best_cfg['batch'], n_roi_features, config.window_size, shuffle=False)
        
        # Compute class weights
        class_weight = compute_class_weights_efficient(train_files)
        
        # Build and compile model
        model = build_tcn_transformer(best_cfg, use_mixed_precision)
        
        optimizer = optimizers.Adam(best_cfg['lr'], clipnorm=1.0)  # Gradient clipping
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[tf.keras.metrics.AUC(name='auroc'),
                     tf.keras.metrics.BinaryAccuracy()]
        )
        
        # Print model summary for first fold
        if fold == 0:
            model.summary()
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor="val_auroc", patience=8, mode="max", restore_best_weights=True),
            callbacks.ModelCheckpoint(
                f"{config.model_dir}/fold{fold+1}_best.h5", 
                save_best_only=True, 
                monitor="val_auroc", 
                mode="max"
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.6, 
                patience=4, 
                min_lr=1e-7, 
                verbose=1
            ),
            callbacks.CSVLogger(f"{config.log_dir}/train_log_fold{fold+1}.csv")
        ]
        
        # Train model
        logger.info("Starting training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            class_weight=class_weight,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        y_true, y_pred = evaluate_model_efficiently(model, val_ds)
        
        # Calculate metrics
        auroc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred > 0.5)
        
        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 81)
        best_acc = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            acc_thresh = accuracy_score(y_true, y_pred > thresh)
            if acc_thresh > best_acc:
                best_acc = acc_thresh
                best_threshold = thresh
        
        # Save predictions and results
        results = {
            'fold': fold + 1,
            'auroc': float(auroc),
            'accuracy': float(acc),
            'best_threshold': float(best_threshold),
            'best_accuracy': float(best_acc),
            'config': best_cfg,
            'n_train': len(train_files),
            'n_val': len(val_files)
        }
        fold_results.append(results)
        
        np.savez(
            f"{config.log_dir}/fold{fold+1}_predictions.npz", 
            y_true=y_true, 
            y_pred=y_pred,
            **results
        )
        
        logger.info(f"Fold {fold+1} results:")
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  Accuracy (0.5): {acc:.4f}")
        logger.info(f"  Best Accuracy: {best_acc:.4f} (threshold: {best_threshold:.3f})")
        
        # Save fold configuration
        with open(f"{config.log_dir}/fold{fold+1}_config.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Clean up for next fold
        del model
        del train_ds
        del val_ds
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Calculate overall results
    logger.info("\n" + "="*50)
    logger.info("OVERALL RESULTS")
    logger.info("="*50)
    
    aurocs = [r['auroc'] for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    best_accs = [r['best_accuracy'] for r in fold_results]
    
    logger.info(f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    logger.info(f"Accuracy (0.5): {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info(f"Best Accuracy: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    
    # Save overall results
    overall_results = {
        'mean_auroc': float(np.mean(aurocs)),
        'std_auroc': float(np.std(aurocs)),
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_best_accuracy': float(np.mean(best_accs)),
        'std_best_accuracy': float(np.std(best_accs)),
        'fold_results': fold_results,
        'dataset_info': {
            'total_samples': int(total_samples),
            'n_roi_features': int(n_roi_features),
            'n_files': len(valid_files)
        }
    }
    
    with open(f"{config.log_dir}/overall_results.json", 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {config.log_dir}")

def create_inference_model(model_path, config_path=None):
    """Create model for inference with proper configuration"""
    try:
        # Load configuration if available
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                model_config = config_data.get('config', {})
        else:
            # Default configuration
            model_config = {
                'blocks': 4,
                'filters': 48,
                'dense_dim': 96,
                'dropout': 0.2,
                'n_roi_features': 15,
                'window_size': 150
            }
        
        # Build model architecture
        model = build_tcn_transformer(model_config)
        
        # Load weights
        model.load_weights(model_path)
        
        logger.info(f"Inference model loaded from {model_path}")
        return model, model_config
        
    except Exception as e:
        logger.error(f"Failed to load inference model: {e}")
        return None, None

def predict_single_window(model, roi_features, window_mask):
    """Predict on a single window of ROI features"""
    try:
        # Ensure proper shapes
        if roi_features.ndim == 2:
            roi_features = np.expand_dims(roi_features, 0)  # Add batch dimension
        if window_mask.ndim == 1:
            window_mask = np.expand_dims(window_mask, 0)  # Add batch dimension
        
        # Make prediction
        prediction = model([roi_features, window_mask], training=False)
        return float(prediction.numpy().squeeze())
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 0.5  # Return neutral prediction on error

# --- Utility functions for analysis ---

def analyze_feature_importance(model, val_dataset, n_samples=100):
    """Analyze which ROI features are most important (simple ablation)"""
    logger.info("Analyzing feature importance...")
    
    # Get some validation samples
    sample_count = 0
    baseline_preds = []
    feature_ablation_preds = {i: [] for i in range(15)}  # 15 ROI features
    
    for batch in val_dataset:
        if sample_count >= n_samples:
            break
            
        roi_features, window_mask, labels = batch
        batch_size = roi_features.shape[0]
        
        # Baseline predictions
        baseline = model([roi_features, window_mask], training=False)
        baseline_preds.extend(baseline.numpy().squeeze())
        
        # Feature ablation - zero out each feature one by one
        for feature_idx in range(15):
            roi_ablated = roi_features.numpy().copy()
            roi_ablated[:, :, feature_idx] = 0  # Zero out feature
            
            ablated_preds = model([roi_ablated, window_mask], training=False)
            feature_ablation_preds[feature_idx].extend(ablated_preds.numpy().squeeze())
        
        sample_count += batch_size
    
    # Calculate importance as change in prediction
    feature_names = []
    rois = ['left_cheek', 'right_cheek', 'forehead', 'chin', 'nose']
    methods = ['chrom', 'pos', 'ica']
    
    for roi in rois:
        for method in methods:
            feature_names.append(f"{roi}_{method}")
    
    importance_scores = {}
    baseline_preds = np.array(baseline_preds)
    
    for i, feature_name in enumerate(feature_names):
        ablated = np.array(feature_ablation_preds[i])
        # Importance = how much prediction changes when feature is removed
        importance = np.mean(np.abs(baseline_preds - ablated))
        importance_scores[feature_name] = importance
    
    # Sort by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("Feature importance ranking:")
    for i, (feature, score) in enumerate(sorted_features[:10]):  # Top 10
        logger.info(f"{i+1:2d}. {feature:20s}: {score:.4f}")
    
    return importance_scores

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise