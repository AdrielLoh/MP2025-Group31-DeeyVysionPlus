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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def setup_gpu():
    """Setup GPU with proper error handling"""
    try:
        physical_devices = tf.config.list_physical_devices('GPU')  # Fix #22: Use non-deprecated API
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"Found {len(physical_devices)} GPU(s), enabled memory growth")
        else:
            logger.info("No GPU devices found, using CPU")
    except Exception as e:
        logger.warning(f"GPU setup failed: {e}")

def setup_mixed_precision():
    """Setup mixed precision with proper configuration"""
    try:
        # Fix #6: Use mixed precision more carefully
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled")
        return True
    except Exception as e:
        logger.warning(f"Mixed precision setup failed: {e}, using float32")
        return False

# --- Configuration ---
def get_config():
    """Get configuration from command line or defaults"""
    parser = argparse.ArgumentParser(description='Train physiological deepfake detection model')
    parser.add_argument('--data_dir', type=str, default='preprocessed_data', 
                       help='Directory containing preprocessed HDF5 files')
    parser.add_argument('--model_dir', type=str, default='outputs/models',
                       help='Directory to save trained models')
    parser.add_argument('--log_dir', type=str, default='outputs/logs',
                       help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--optuna_trials', type=int, default=10,
                       help='Number of Optuna trials for hyperparameter search')
    parser.add_argument('--optuna_timeout', type=int, default=3600,
                       help='Optuna timeout in seconds')
    parser.add_argument('--window_size', type=int, default=150,
                       help='Expected window size')
    parser.add_argument('--skip_optuna', action='store_true',
                       help='Skip hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    return args

# --- Data Loader ---

def extract_label_from_filename(filename):
    """Extract label from filename with fallback to HDF5 attributes (FIXED)"""
    filename_path = filename
    filename = os.path.basename(filename).lower()
    
    # Primary method: check filename patterns
    if 'fake' in filename or 'deepfake' in filename or 'synthetic' in filename:
        return 1
    elif 'real' in filename or 'authentic' in filename or 'genuine' in filename:
        return 0
    else:
        # CRITICAL FIX: Fallback to HDF5 attributes if filename is ambiguous
        try:
            with h5py.File(filename_path, 'r') as f:
                if 'dataset_label' in f.attrs:
                    label_str = f.attrs['dataset_label']
                    # FIX: Safer string handling
                    if isinstance(label_str, (bytes, np.bytes_)):
                        label_str = label_str.decode('utf-8')
                    return 1 if 'fake' in str(label_str).lower() else 0
                elif 'is_fake' in f.attrs:
                    return int(f.attrs['is_fake'])
        except Exception as e:
            logger.warning(f"Could not read label from {filename_path}: {e}")
        
        # Try to infer from directory structure
        parts = filename.split('_')
        for part in parts:
            if 'fake' in part or 'real' in part:
                return 1 if 'fake' in part else 0
        
        logger.warning(f"Could not determine label for {filename}, assuming real (0)")
        return 0

def extract_video_id(filename):
    """Extract video ID for group-based splitting with HDF5 support (FIXED)"""
    # CRITICAL FIX: Try to get original video name from HDF5 first
    try:
        with h5py.File(filename, 'r') as f:
            # Get video IDs from all videos in the file
            video_ids = []
            for vid_name in f.keys():
                vid_group = f[vid_name]
                if 'original_filename' in vid_group.attrs:
                    orig_name = vid_group.attrs['original_filename']
                    if isinstance(orig_name, (bytes, np.bytes_)):
                        orig_name = orig_name.decode('utf-8')
                    # Remove extension and use as ID
                    video_id = os.path.splitext(orig_name)[0]
                    video_ids.append(video_id)
                else:
                    video_ids.append(vid_name)
            
            # Return combined ID for this file
            return '_'.join(sorted(set(video_ids)))
            
    except Exception as e:
        logger.warning(f"Could not extract video ID from HDF5 {filename}: {e}")
    
    # Fallback to filename-based extraction
    basename = os.path.basename(filename)
    if '_batch' in basename:
        parts = basename.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[2:]).replace('.h5', '')
    return basename.replace('.h5', '')

def validate_hdf5_file(filepath):
    """Validate HDF5 file structure (IMPROVED)"""
    try:
        with h5py.File(filepath, 'r') as f:
            if len(f.keys()) == 0:
                return False
            
            # Check first video/window structure
            for vid_name in f.keys():
                vid_group = f[vid_name]
                if len(vid_group.keys()) == 0:
                    continue
                    
                for win_name in vid_group.keys():
                    win_group = vid_group[win_name]
                    
                    # Check required groups
                    if 'global' not in win_group or 'multi_roi' not in win_group:
                        return False
                    
                    # Check required datasets
                    required_datasets = ['face_mask', 'window_mask']
                    for dataset in required_datasets:
                        if dataset not in win_group:
                            return False
                    
                    # CRITICAL FIX: Validate multi_roi structure
                    roi_group = win_group['multi_roi']
                    if len(roi_group.keys()) == 0:
                        return False
                    
                    # Check that each ROI has datasets
                    for roi_name in roi_group.keys():
                        roi_subgroup = roi_group[roi_name]
                        if len(roi_subgroup.keys()) == 0:
                            return False
                    
                    return True  # If we get here, structure is valid
        return False
    except Exception as e:
        logger.warning(f"File validation failed for {filepath}: {e}")
        return False

def load_window(h5_window, expected_window_size=150):
    """Load window data with validation (FIXED)"""
    try:
        # Load global features
        g_dict = h5_window['global']
        g_feats = []
        for k in sorted(g_dict.keys()):  # Consistent ordering
            data = g_dict[k][()]
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.warning(f"Invalid values in global feature {k}")
                data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
            g_feats.append(data)
        
        if not g_feats:
            raise ValueError("No global features found")
            
        X_global = np.stack(g_feats, axis=-1)
        
        # CRITICAL FIX: Load ROI features with proper structure
        if 'multi_roi' not in h5_window:
            raise ValueError("multi_roi group not found")
            
        roi_group = h5_window['multi_roi']
        rois = sorted(roi_group.keys())  # Consistent ordering
        
        # Collect all ROI features in order
        all_roi_features = []
        
        for roi in rois:
            r_dict = roi_group[roi]
            roi_data = []
            for k in sorted(r_dict.keys()):  # Consistent feature ordering
                data = r_dict[k][()]
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    logger.warning(f"Invalid values in ROI {roi} feature {k}")
                    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                roi_data.append(data)
            
            if roi_data:
                # Stack features for this ROI
                roi_features = np.stack(roi_data, axis=-1)  # (time, n_features_for_this_roi)
                all_roi_features.append(roi_features)
        
        if not all_roi_features:
            raise ValueError("No ROI features found")
        
        # CRITICAL FIX: Flatten ROI features correctly
        # Concatenate all ROI features along the feature dimension
        X_roi_flat = np.concatenate(all_roi_features, axis=-1)  # (time, total_roi_features)
        
        # Load masks
        window_mask = h5_window['window_mask'][()]
        
        # Validate dimensions
        if X_global.shape[0] != expected_window_size:
            logger.warning(f"Global features shape mismatch: {X_global.shape[0]} != {expected_window_size}")
        if X_roi_flat.shape[0] != expected_window_size:
            logger.warning(f"ROI features shape mismatch: {X_roi_flat.shape[0]} != {expected_window_size}")
        if window_mask.shape[0] != expected_window_size:
            logger.warning(f"Window mask shape mismatch: {window_mask.shape[0]} != {expected_window_size}")
        
        return (X_global.astype(np.float32),
                X_roi_flat.astype(np.float32),
                window_mask.astype(np.float32))
        
    except Exception as e:
        logger.error(f"Failed to load window: {e}")
        raise

def count_samples_and_get_shapes(file_paths):
    """Count total samples and get feature shapes (FIXED)"""
    total_samples = 0
    n_g, n_roi = None, None
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
                        
                        # CRITICAL FIX: Get shapes from first valid window more carefully
                        if n_g is None or n_roi is None:
                            g_dict = win_group['global']
                            roi_dict = win_group['multi_roi']
                            n_g = len(g_dict.keys())
                            
                            # CRITICAL FIX: Count total ROI features correctly
                            n_roi_features = 0
                            for roi_name in roi_dict.keys():
                                roi_subgroup = roi_dict[roi_name]
                                n_roi_features += len(roi_subgroup.keys())
                            n_roi = n_roi_features
                        
                        total_samples += 1
                        labels.append(file_label)
        except Exception as e:
            logger.warning(f"Error reading {filepath}: {e}")
            continue
    
    logger.info(f"Found {total_samples} samples, {n_g} global features, {n_roi} ROI features")
    return total_samples, n_g, n_roi, np.array(labels)

def create_data_generator(file_paths, batch_size, n_g, n_roi, window_size=150, shuffle=True, infinite=True):
    """Create efficient data generator (fix #5, #4)"""
    def generator():
        file_indices = list(range(len(file_paths)))
        
        while True:
            if shuffle:
                np.random.shuffle(file_indices)
            
            batch_global = []
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
                                    X_global, X_roi_flat, window_mask = load_window(win_group, window_size)
                                    
                                    batch_global.append(X_global)
                                    batch_roi.append(X_roi_flat)
                                    batch_mask.append(window_mask)
                                    batch_labels.append(file_label)
                                    
                                    if len(batch_global) == batch_size:
                                        yield (np.stack(batch_global),
                                               np.stack(batch_roi),
                                               np.stack(batch_mask),
                                               np.array(batch_labels))
                                        
                                        batch_global = []
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
            if batch_global and len(batch_global) > 0:
                yield (np.stack(batch_global),
                       np.stack(batch_roi),
                       np.stack(batch_mask),
                       np.array(batch_labels))
            
            if not infinite:
                break
    
    return generator

def make_dataset(file_paths, batch_size, n_g, n_roi, window_size=150, shuffle=True):
    """Create TensorFlow dataset (fix #19)"""
    generator_fn = create_data_generator(file_paths, batch_size, n_g, n_roi, window_size, shuffle, infinite=False)
    
    output_signature = (
        tf.TensorSpec(shape=(None, window_size, n_g), dtype=tf.float32),
        tf.TensorSpec(shape=(None, window_size, n_roi), dtype=tf.float32),
        tf.TensorSpec(shape=(None, window_size), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    
    ds = tf.data.Dataset.from_generator(generator_fn, output_signature=output_signature)
    
    # Adaptive shuffle buffer based on dataset size (fix #19)
    if shuffle:
        buffer_size = min(max(batch_size * 10, 100), 2048)
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)
    
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- Model ---

def tcn_block(filters, dilation, dropout_rate=0.1):
    """TCN block with configurable dropout"""
    def block(x):
        y = layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation)(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y
    return block

def build_feature_encoder(input_len, input_channels, target_channels, num_blocks, base_filters):
    """Build encoder that projects input to target channels (fix #3, #9)"""
    inp = layers.Input((input_len, input_channels))
    
    # Project to target channels if needed
    if input_channels != target_channels:
        x = layers.Dense(target_channels)(inp)
    else:
        x = inp
    
    # TCN blocks
    for b in range(num_blocks):
        y = tcn_block(base_filters, 2 ** b)(x)
        if x.shape[-1] == y.shape[-1]:
            x = layers.Add()([x, y])
        else:
            # Project x to match y's channels
            x_proj = layers.Dense(y.shape[-1])(x)
            x = layers.Add()([x_proj, y])
    
    return models.Model(inp, x, name=f'Encoder_{input_channels}to{target_channels}')

def build_dual_stream(cfg, use_mixed_precision=False):
    """Build dual-stream model with proper channel handling"""
    window_size = cfg.get('window_size', 150)
    
    g_in = layers.Input((window_size, cfg['n_g']), name='global_in')
    roi_in = layers.Input((window_size, cfg['n_roi']), name='roi_in')
    mask = layers.Input((window_size,), name='mask_in')
    
    # Use common target channels for both streams
    target_channels = cfg['filters']
    
    # Build separate encoders for each stream (fix #3)
    g_encoder = build_feature_encoder(window_size, cfg['n_g'], target_channels, cfg['blocks'], cfg['filters'])
    roi_encoder = build_feature_encoder(window_size, cfg['n_roi'], target_channels, cfg['blocks'], cfg['filters'])
    
    g_feat = g_encoder(g_in)
    roi_feat = roi_encoder(roi_in)
    
    def masked_gap(args):
        """Masked Global Average Pooling with better numerical stability (fix #15)"""
        f, m = args
        m = tf.cast(tf.expand_dims(m, -1), f.dtype)
        # Use larger epsilon for better stability
        return tf.reduce_sum(f * m, 1) / tf.clip_by_value(tf.reduce_sum(m, 1, keepdims=True), 1e-3, tf.float32.max)
    
    g_vec = layers.Lambda(masked_gap)([g_feat, mask])
    roi_vec = layers.Lambda(masked_gap)([roi_feat, mask])
    
    # Fusion layer
    fusion = layers.Concatenate()([g_vec, roi_vec])
    fusion = layers.Dense(cfg['att_dim'])(fusion)
    fusion = layers.ReLU()(fusion)
    fusion = layers.Dropout(0.2)(fusion)
    
    # Output layer with proper dtype handling for mixed precision (fix #6)
    if use_mixed_precision:
        out = layers.Dense(1, activation='sigmoid', dtype='float32')(fusion)
    else:
        out = layers.Dense(1, activation='sigmoid')(fusion)
    
    return models.Model([g_in, roi_in, mask], out)

# --- Training utilities ---

def compute_class_weights_efficient(file_paths):
    """Efficiently compute class weights without loading all data (fix #4)"""
    n_real = 0
    n_fake = 0
    
    for filepath in file_paths:
        label = extract_label_from_filename(filepath)
        
        # Count windows in this file
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
    """Efficiently evaluate model and collect predictions (fix #10, #16)"""
    y_true = []
    y_pred = []
    
    for batch in val_dataset:
        Xg, Xr, Xm, yb = batch
        # Use direct call instead of predict() for efficiency (fix #10)
        preds = model([Xg, Xr, Xm], training=False)
        y_true.extend(yb.numpy())
        y_pred.extend(preds.numpy().squeeze())
    
    return np.array(y_true), np.array(y_pred)

# --- Optuna objective ---

def objective(trial, train_files, val_files, n_g, n_roi, window_size, use_mixed_precision):
    """Optuna objective with proper cleanup (fix #8)"""
    try:
        cfg = {
            'blocks': trial.suggest_int('blocks', 3, 6),
            'filters': trial.suggest_categorical('filters', [64, 96, 128]),
            'att_dim': trial.suggest_categorical('att_dim', [64, 128, 256]),
            'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'batch': trial.suggest_categorical('batch', [16, 32]),
            'n_g': n_g,
            'n_roi': n_roi,
            'window_size': window_size
        }
        
        # Create datasets
        train_ds = make_dataset(train_files, cfg['batch'], n_g, n_roi, window_size, shuffle=True)
        val_ds = make_dataset(val_files, cfg['batch'], n_g, n_roi, window_size, shuffle=False)
        
        # Build model
        model = build_dual_stream(cfg, use_mixed_precision)
        
        # Compile with proper loss scaling for mixed precision
        optimizer = optimizers.Adam(cfg['lr'])
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=[tf.keras.metrics.AUC(name='auroc')]
        )
        
        # Train with early stopping
        es = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_auroc', mode='max')
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,  # Shorter for hyperparameter search
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
        # Clean up on failure
        tf.keras.backend.clear_session()
        gc.collect()
        return 1.0  # Return worst possible score

# --- Main training function ---

def main():
    # Setup
    config = get_config()
    setup_gpu()
    use_mixed_precision = setup_mixed_precision()
    
    logger.info(f"Starting training with config: {vars(config)}")
    
    # Find and validate files (fix #14, #23)
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
    
    # Get dataset information
    total_samples, n_g, n_roi, file_labels = count_samples_and_get_shapes(valid_files)
    logger.info(f"Dataset: {total_samples} samples, {n_g} global features, {n_roi} ROI features")
    
    # Extract video IDs for group-based splitting (fix #7)
    video_ids = [extract_video_id(f) for f in valid_files]
    
    # Use GroupKFold to prevent data leakage (fix #7)
    if len(set(video_ids)) >= config.folds:
        kfold = GroupKFold(n_splits=config.folds)
        splits = list(kfold.split(valid_files, file_labels, video_ids))
    else:
        logger.warning("Not enough unique videos for GroupKFold, using StratifiedKFold")
        kfold = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=42)
        splits = list(kfold.split(valid_files, file_labels))
    
    # Training loop
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n===== Fold {fold+1}/{config.folds} =====")
        
        train_files = [valid_files[i] for i in train_idx]
        val_files = [valid_files[i] for i in val_idx]
        
        logger.info(f"Train: {len(train_files)} files | Val: {len(val_files)} files")
        
        # Hyperparameter optimization
        if not config.skip_optuna:
            logger.info("Starting hyperparameter optimization...")
            
            def fold_objective(trial):
                return objective(trial, train_files, val_files, n_g, n_roi, config.window_size, use_mixed_precision)
            
            study = optuna.create_study(direction='minimize')
            study.optimize(fold_objective, n_trials=config.optuna_trials, timeout=config.optuna_timeout)
            
            best_cfg = {
                **study.best_params,
                'n_g': n_g,
                'n_roi': n_roi,
                'window_size': config.window_size
            }
            logger.info(f"Best config for fold {fold+1}: {best_cfg}")
        else:
            # Use default configuration
            best_cfg = {
                'blocks': 4,
                'filters': 96,
                'att_dim': 128,
                'lr': 1e-3,
                'batch': 32,
                'n_g': n_g,
                'n_roi': n_roi,
                'window_size': config.window_size
            }
            logger.info(f"Using default config for fold {fold+1}: {best_cfg}")
        
        # Create datasets
        train_ds = make_dataset(train_files, best_cfg['batch'], n_g, n_roi, config.window_size, shuffle=True)
        val_ds = make_dataset(val_files, best_cfg['batch'], n_g, n_roi, config.window_size, shuffle=False)
        
        # Compute class weights
        class_weight = compute_class_weights_efficient(train_files)
        
        # Build and compile model
        model = build_dual_stream(best_cfg, use_mixed_precision)
        
        optimizer = optimizers.Adam(best_cfg['lr'])
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=[tf.keras.metrics.AUC(name='auroc'),
                     tf.keras.metrics.BinaryAccuracy()]
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor="val_auroc", patience=7, mode="max", restore_best_weights=True),
            callbacks.ModelCheckpoint(
                f"{config.model_dir}/fold{fold+1}.h5", 
                save_best_only=True, 
                monitor="val_auroc", 
                mode="max"
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=3, 
                min_lr=1e-6, 
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
        
        # Save predictions for threshold tuning
        np.savez(
            f"{config.log_dir}/fold{fold+1}_valpreds.npz", 
            y_true=y_true, 
            y_pred=y_pred,
            config=best_cfg
        )
        
        # Log results
        auroc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred > 0.5)
        
        logger.info(f"Fold {fold+1} results: AUROC={auroc:.4f}, Accuracy={acc:.4f}")
        
        # Clean up for next fold
        del model
        del train_ds
        del val_ds
        tf.keras.backend.clear_session()
        gc.collect()
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()