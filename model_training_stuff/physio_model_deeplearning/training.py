import os
import glob
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision, regularizers
from tqdm import tqdm
import logging
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import json
import pandas as pd
import random
from tensorflow.keras.utils import register_keras_serializable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Setup for ROCm (WSL2 Ubuntu) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def setup_mixed_precision():
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled (mixed_float16)")
        return True
    except Exception as e:
        logger.warning(f"Mixed precision setup failed: {e}, using float32")
        return False
    
# --- Filter based on original video name criteria ---
def keep_row(row):
    video_id = str(row['video_id']).lower()
    if row['label'] == 1:
        # Fake: only celeb-df2_ or ffpp_
        return ("celeb-df2_" in video_id) or ("ffpp_" in video_id)
    else:
        # Real: only celeb-df2_, ffpp_, or dfdc_
        return ("celeb-df2_" in video_id) or ("ffpp_" in video_id) or ("dfdc_" in video_id) or ("deeperforensics_" in video_id)

def get_config():
    """Get configuration optimized for ROCm with 16GB VRAM"""
    parser = argparse.ArgumentParser(description='Train physiological deepfake detection model')
    parser.add_argument('--data_dir', type=str, default='/root/model_training/cache/batches/physio-deep-v1/for-training',
                       help='Directory containing preprocessed HDF5 files')
    parser.add_argument('--model_dir', type=str, default='/root/model_training/physiological-model/deep-learning-1-1',
                       help='Directory to save trained models')
    parser.add_argument('--log_dir', type=str, default='/root/model_training/physiological-model/deep-learning-1-1/logs',
                       help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--optuna_trials', type=int, default=15,
                       help='Number of Optuna trials for hyperparameter search')
    parser.add_argument('--optuna_timeout', type=int, default=7200,
                       help='Optuna timeout in seconds')
    parser.add_argument('--window_size', type=int, default=150,
                       help='Expected window size')
    parser.add_argument('--skip_optuna', action='store_true',
                       help='Skip hyperparameter optimization')
    parser.add_argument('--batch_size', type=int, default=8,  # Suitable for 16GB VRAM, adjust if needed
                       help='Batch size (optimized for 16GB VRAM)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--eval_only', action='store_true', help='Skip training and run evaluation only')
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args

# --- Updated Data Loader for New Feature Structure ---
def extract_label_from_filename(filename):
    filename_path = filename
    filename = os.path.basename(filename).lower()
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
                    required_datasets = ['window_mask']
                    for dataset in required_datasets:
                        if dataset not in win_group:
                            return False
                    if 'multi_roi' not in win_group:
                        return False
                    roi_group = win_group['multi_roi']
                    available_rois = list(roi_group.keys())
                    if len(available_rois) == 0:
                        return False
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
    try:
        if 'multi_roi' not in h5_window:
            raise ValueError("multi_roi group not found")
        roi_group = h5_window['multi_roi']
        available_rois = sorted(roi_group.keys())
        all_roi_features = []
        for roi in available_rois:
            r_dict = roi_group[roi]
            roi_data = []
            for feature_name in ['chrom', 'pos', 'ica']:
                if feature_name in r_dict:
                    data = r_dict[feature_name][()]
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        logger.warning(f"Invalid values in ROI {roi} feature {feature_name}")
                        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                    roi_data.append(data)
                else:
                    logger.warning(f"Missing feature {feature_name} in ROI {roi}")
                    roi_data.append(np.zeros(expected_window_size, dtype=np.float32))
            if roi_data:
                roi_features = np.stack(roi_data, axis=-1)
                all_roi_features.append(roi_features)
        if not all_roi_features:
            raise ValueError("No ROI features found")
        X_roi_combined = np.concatenate(all_roi_features, axis=-1)

        # ===== Normalize all features to mean 0, std 1 per batch =====
        mu = np.mean(X_roi_combined, axis=0, keepdims=True)
        sigma = np.std(X_roi_combined, axis=0, keepdims=True) + 1e-8
        X_roi_combined = (X_roi_combined - mu) / sigma

        window_mask = h5_window['window_mask'][()]
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
                        if n_roi_features is None:
                            roi_dict = win_group['multi_roi']
                            n_roi_features = 0
                            for roi_name in roi_dict.keys():
                                roi_subgroup = roi_dict[roi_name]
                                n_roi_features += len([k for k in roi_subgroup.keys() if k in ['chrom', 'pos', 'ica']])
                        total_samples += 1
                        labels.append(file_label)
        except Exception as e:
            logger.warning(f"Error reading {filepath}: {e}")
            continue
    if n_roi_features is None:
        n_roi_features = 15  # Default expected
    logger.info(f"Found {total_samples} samples, {n_roi_features} ROI features total")
    return total_samples, n_roi_features, np.array(labels)

def build_window_metadata(hdf5_files):
    rows = []
    for h5_path in hdf5_files:
        label = extract_label_from_filename(h5_path)
        try:
            with h5py.File(h5_path, 'r') as f:
                for group_name in f.keys():
                    g = f[group_name]
                    video_id = g.attrs.get('original_filename', group_name)
                    if isinstance(video_id, bytes):
                        video_id = video_id.decode()
                    for win_name in g.keys():
                        if not win_name.startswith('window_'):
                            continue
                        rows.append({
                            'file_path': h5_path,
                            'group_name': group_name,
                            'window_name': win_name,
                            'label': label,
                            'video_id': video_id,
                        })
        except Exception as e:
            print(f"Error reading {h5_path}: {e}")
    return pd.DataFrame(rows)

def window_level_data_generator(df, batch_size, window_size=150, n_roi_features=15, shuffle=True, infinite=True, augment=False):
    indices = df.index.to_numpy()
    config = get_config()
    if shuffle:
        df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    indices = df.index.to_numpy()
    batch_roi = []
    batch_mask = []
    batch_labels = []
    for idx in indices:
        row = df.loc[idx]
        with h5py.File(row['file_path'], 'r') as f:
            win_group = f[row['group_name']][row['window_name']]
            X_roi_combined, window_mask = load_window_new_format(win_group, window_size)
            if augment and random.random() < 0.4:
                X_roi_combined += np.random.normal(0, 0.02, X_roi_combined.shape).astype(np.float32)
                mask_prob = 0.1
                mask = np.random.binomial(1, mask_prob, X_roi_combined.shape).astype(bool)
                X_roi_combined[mask] = 0.0
            batch_roi.append(X_roi_combined)
            batch_mask.append(window_mask)
            batch_labels.append(row['label'])
        if len(batch_roi) == batch_size:
            yield (
                (np.stack(batch_roi), np.stack(batch_mask)),
                np.array(batch_labels).astype('float32').reshape(-1, 1)
            )
            batch_roi, batch_mask, batch_labels = [], [], []
    if batch_roi:
        yield (
            (np.stack(batch_roi), np.stack(batch_mask)),
            np.array(batch_labels).astype('float32').reshape(-1, 1)
        )
@register_keras_serializable()
def masked_gap(args):
    f, m = args
    seq_len = tf.shape(f)[1]
    m = m[:, :seq_len]
    m = tf.cast(tf.expand_dims(m, -1), f.dtype)
    num = tf.reduce_sum(f * m, axis=1)
    denom = tf.reduce_sum(m, axis=1)
    pooled = num / tf.clip_by_value(denom, 1e-3, tf.float32.max)
    return pooled

def build_tcn_transformer(cfg, use_mixed_precision=False):
    window_size = cfg.get('window_size', 150)
    n_roi_features = cfg['n_roi_features']
    tcn_channels = cfg.get('filters', 48)
    ff_dim = cfg.get('dense_dim', 96)
    num_heads = 4 # TUNEABLE
    attn_dropout = cfg.get('dropout', 0.2)
    tcn_blocks = [2 ** i for i in range(cfg['blocks'])]

    roi_in = layers.Input((window_size, n_roi_features), name='roi_in')
    mask = layers.Input((window_size,), name='mask_in')

    x = roi_in
    for d in tcn_blocks:
        res = x
        x = layers.Conv1D(tcn_channels, 3, padding="causal", dilation_rate=d, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(tcn_channels, 3, padding="causal", dilation_rate=d, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        if int(res.shape[-1]) != tcn_channels:
            res = layers.Conv1D(tcn_channels, 1, padding="same", kernel_regularizer=regularizers.l2(1e-4))(res)
        x = layers.Add()([res, x])
        x = layers.Activation("relu")(x)

    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(ff_dim, 1, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    seq_len = x.shape[1]
    pos_indices = tf.range(seq_len)
    pos_emb_layer = layers.Embedding(input_dim=seq_len, output_dim=ff_dim)
    pos_emb = pos_emb_layer(pos_indices)
    pos_emb = tf.expand_dims(pos_emb, 0)
    x = layers.Add()([x, pos_emb])

    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=ff_dim // num_heads,
        dropout=attn_dropout
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ffn = layers.Dense(ff_dim, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    ffn = layers.Dense(ff_dim, kernel_regularizer=regularizers.l2(1e-4))(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    pooled = layers.Lambda(masked_gap, name='masked_pooling')([x, mask])

    dense = layers.Dense(ff_dim, kernel_regularizer=regularizers.l2(1e-4))(pooled)
    dense = layers.ReLU()(dense)
    dense = layers.Dropout(attn_dropout)(dense)
    dense = layers.Dense(ff_dim // 2, kernel_regularizer=regularizers.l2(1e-4))(dense)
    dense = layers.ReLU()(dense)
    dense = layers.Dropout(attn_dropout)(dense)
    if use_mixed_precision:
        out = layers.Dense(1, activation='sigmoid', dtype='float32', name='output', kernel_regularizer=regularizers.l2(1e-4))(dense)
    else:
        out = layers.Dense(1, activation='sigmoid', name='output', kernel_regularizer=regularizers.l2(1e-4))(dense)
    model = models.Model([roi_in, mask], out, name='TCN_Transformer')
    return model

def evaluate_model_efficiently(model, val_dataset):
    y_true = []
    y_pred = []
    batch_num = 0
    for batch in tqdm(val_dataset, desc="Evaluating val batches"):
        (Xr, Xm), yb = batch
        Xr_np = Xr.numpy()
        Xm_np = Xm.numpy()
        yb_np = yb.numpy()
        
        # --- Debugging info ---
        min_val, max_val = np.nanmin(Xr_np), np.nanmax(Xr_np)
        any_nan = np.any(np.isnan(Xr_np))
        any_inf = np.any(np.isinf(Xr_np))
        label_nan = np.any(np.isnan(yb_np))
        label_inf = np.any(np.isinf(yb_np))
        
        if batch_num < 3 or any_nan or any_inf or label_nan or label_inf or min_val < -1e3 or max_val > 1e3:
            print(f"[VAL][Batch {batch_num}] Xr min/max: {min_val:.5f}/{max_val:.5f}, NaN: {any_nan}, Inf: {any_inf}")
            print(f"[VAL][Batch {batch_num}] Labels unique: {np.unique(yb_np)}, NaN: {label_nan}, Inf: {label_inf}")
        
        if any_nan or any_inf or label_nan or label_inf:
            print(f"[VAL][Batch {batch_num}] WARNING: Skipping batch due to NaN/Inf in data or labels!")
            continue  # Optionally skip or raise error
        
        preds = model([Xr, Xm], training=False)
        y_true.extend(yb_np)
        y_pred.extend(np.ravel(preds.numpy()))
        batch_num += 1
    return np.array(y_true), np.array(y_pred)

def find_youden_j_threshold(y_true, y_pred, thresholds=np.linspace(0.01, 0.99, 99)):
    best_j = -np.inf
    best_thresh = 0.5
    for thresh in thresholds:
        preds_bin = (y_pred >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds_bin, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        j = sensitivity + specificity - 1
        if j > best_j:
            best_j = j
            best_thresh = thresh
    return best_thresh, best_j

def main():
    config = get_config()
    set_seed(config.seed)
    with open(f"{config.log_dir}/global_config.json", 'w') as f:
        json.dump(vars(config), f, indent=2)

    # Standard TensorFlow GPU memory growth for ROCm
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info("Enabled GPU memory growth.")
        except Exception as e:
            logger.warning(f"Failed to set GPU memory growth: {e}")

    use_mixed_precision = setup_mixed_precision()
    logger.info(f"Starting training with config: {vars(config)}")
    logger.info(f"Mixed precision: {use_mixed_precision}")

    if not os.path.exists(config.data_dir):
        logger.error(f"Data directory does not exist: {config.data_dir}")
        return

    all_files = sorted(glob.glob(os.path.join(config.data_dir, "*.h5")))
    if not all_files:
        logger.error(f"No HDF5 files found in {config.data_dir}")
        return

    logger.info(f"Found {len(all_files)} HDF5 files")

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

    total_samples, n_roi_features, _ = count_samples_and_get_shapes(valid_files)

    logger.info(f"Dataset: {total_samples} samples, {n_roi_features} ROI features")

    # Build and Filter dataframes
    df = build_window_metadata(valid_files)
    filter_dataframes = True # ----- TOGGLE FILTERING ON / OFF
    if filter_dataframes:
        df = df[df.apply(keep_row, axis=1)].reset_index(drop=True)
        print("Filtered windows by source:")
        print(df.groupby("label")["video_id"].apply(lambda x: x.str[:30].unique()).to_dict())
        print("Remaining window count by class:", df["label"].value_counts())

    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=config.seed,
    )

    print("--- LABEL COUNTS ---", flush=True)
    print(f"Train windows: {len(train_df)}, Validation windows: {len(val_df)}")
    print("Train label counts:", train_df['label'].value_counts())
    print("Val label counts:", val_df['label'].value_counts())

    output_signature = (
        (
            tf.TensorSpec(shape=(None, config.window_size, n_roi_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, config.window_size), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: window_level_data_generator(
            train_df,
            batch_size=config.batch_size,
            window_size=config.window_size,
            n_roi_features=n_roi_features,
            shuffle=True,
            infinite=True,
            augment=True
        ),
        output_signature=output_signature
    ).repeat().prefetch(2)

    val_ds = tf.data.Dataset.from_generator(
        lambda: window_level_data_generator(
            val_df,
            batch_size=config.batch_size,
            window_size=config.window_size,
            n_roi_features=n_roi_features,
            shuffle=False,
            infinite=False,
            augment=False
        ),
        output_signature=output_signature
    ).prefetch(2)

    # After making train_ds and val_ds
    for (x, m), y in train_ds.take(1):
        print("X batch min/max:", np.min(x.numpy()), np.max(x.numpy()))
        print("Y batch unique:", np.unique(y.numpy()))

    best_cfg = {
        'blocks': 6,
        'filters': 64, # 48
        'dense_dim': 128,
        'lr': 0.0001,
        'batch': config.batch_size,
        'dropout': 0.2,
        'n_roi_features': n_roi_features,
        'window_size': config.window_size
    }
    logger.info(f"Using default config: {best_cfg}")

    counts = train_df['label'].value_counts()
    n_real = counts.get(0, 0)
    n_fake = counts.get(1, 0)
    total = n_real + n_fake
    if total == 0 or n_real == 0 or n_fake == 0:
        class_weight = {0: 1.0, 1: 1.0}
    else:
        class_weight = {0: total / (2 * n_real), 1: total / (2 * n_fake)}

    model = build_tcn_transformer(best_cfg, use_mixed_precision)

    # If only evaluating, load model weights and skip training
    if config.eval_only:
        weights_path = "/root/model_training/physiological-model/deep-learning-1-1/checkpoint_best_model.keras"
        if not weights_path or not os.path.isfile(weights_path):
            logger.error(f"Specified weights file does not exist: {weights_path}")
            return
        logger.info(f"Loading model weights from: {weights_path}")
        model.load_weights(weights_path)
        model.save("/mnt/c/Users/Adrie/Documents/GitHub/MP2025-Group31-DeeyVysionPlus/model_training_stuff/physio_model_deeplearning/physio_deep_evaluated.keras")
    else:
        optimizer = optimizers.Adam(best_cfg['lr'], clipnorm=1.0)
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[tf.keras.metrics.AUC(name='auroc'), tf.keras.metrics.BinaryAccuracy()]
        )

        callbacks_list = [
            callbacks.EarlyStopping(monitor="val_auroc", patience=5, mode="max", restore_best_weights=True),
            callbacks.ModelCheckpoint(
                f"{config.model_dir}/checkpoint_best_model.keras",
                save_best_only=True,
                monitor="val_auroc",
                mode="max"
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(f"{config.log_dir}/train_log.csv")
        ]

        # ===== Checking batch data quality issues =====
        for batch in train_ds.take(1):
            (Xr, Xm), yb = batch
            print("Xr min/max:", np.nanmin(Xr.numpy()), np.nanmax(Xr.numpy()))
            print("Xm min/max:", np.nanmin(Xm.numpy()), np.nanmax(Xm.numpy()))
            print("yb unique:", np.unique(yb.numpy()))
            print("Any NaNs in Xr?", np.any(np.isnan(Xr.numpy())))
            print("Any NaNs in Xm?", np.any(np.isnan(Xm.numpy())))
            print("Any NaNs in yb?", np.any(np.isnan(yb.numpy())))

        # ===== Calculate number of steps based on samples =====
        steps_per_epoch = int(np.ceil(len(train_df) / config.batch_size))
        validation_steps = int(np.ceil(len(val_df) / config.batch_size))

        logger.info("Starting training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=callbacks_list,
            verbose=1
        )

        model.save(f"{config.model_dir}/physio_deep_model.keras")

    logger.info("Evaluating model...")
    y_true, y_pred = evaluate_model_efficiently(model, val_ds)
    auroc = roc_auc_score(y_true, y_pred)
    best_threshold, best_j = find_youden_j_threshold(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred > best_threshold)

    results = {
        'auroc': float(auroc),
        'accuracy': float(acc),
        'youden_j': float(best_j),
        'best_threshold': float(best_threshold),
        'config': best_cfg,
        'n_train': len(train_df),
        'n_val': len(val_df)
    }
    np.savez(
        f"{config.log_dir}/predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        **results
    )
    with open(f"{config.log_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
