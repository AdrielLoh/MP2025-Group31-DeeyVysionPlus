import os
import numpy as np
import tensorflow as tf
import random
from glob import glob
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import argparse
from tensorflow.keras.utils import register_keras_serializable

# --- Environment Setup for ROCm (WSL2 Ubuntu) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ===================== Seed All RNGs for Reproducibility =====================
def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ===================== Data Loading =====================
def load_npz_batches(npz_dir):
    files = glob(os.path.join(npz_dir, "*.npz"))
    all_signals, all_masks, all_vids, all_labels = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            all_signals.append(d["signals"])   # [N, 5, 150]
            all_masks.append(d["masks"])       # [N, 150]
            all_vids.append(d["videos"])       # [N,]
            # Either store label in each file or per window (see your preprocessing)
            # We'll read d["label"] (int/str), broadcast for all windows in batch
            if "label" in d:
                label = d["label"]
                if isinstance(label, np.ndarray): label = label.item()
                if isinstance(label, str):
                    label = {'real': 0, 'fake': 1}[label.lower()]
                all_labels.append(np.full(d["signals"].shape[0], int(label)))
            else:
                raise RuntimeError("Each npz file must have 'label' entry")
    X = np.concatenate(all_signals, axis=0)     # [N, 5, 150]
    M = np.concatenate(all_masks, axis=0)       # [N, 150]
    V = np.concatenate(all_vids, axis=0)        # [N,]
    y = np.concatenate(all_labels, axis=0)      # [N,]
    return X, M, V, y

def robust_normalize(X, mask=None, abs_max=20.0):  # very loose for debug
    X_new = []
    mask_new = []
    idx_keep = []
    for i, x in enumerate(X):
        x_norm = np.zeros_like(x)
        for c in range(x.shape[0]):
            arr = x[c]
            m, s = np.mean(arr), np.std(arr)
            if s < 1e-6: s = 1.0
            x_norm[c] = (arr - m) / s
        # Debug print
        if np.any(np.isnan(x_norm)) or np.any(np.isinf(x_norm)):
            print("NaN/Inf at window", i)
            continue
        if np.max(np.abs(x_norm)) > abs_max:
            print("Filtered for abs_max at window", i, "max:", np.max(np.abs(x_norm)))
            continue
        X_new.append(x_norm)
        if mask is not None:
            mask_new.append(mask[i])
        idx_keep.append(i)
    print("Kept", len(X_new), "out of", len(X), "windows")
    if len(X_new) == 0:
        print("All windows filtered! Try increasing abs_max or check your input data.")
        raise RuntimeError("All training data filtered out.")
    X_new = np.stack(X_new, axis=0)
    if mask is not None:
        mask_new = np.stack(mask_new, axis=0)
        return X_new, mask_new, idx_keep
    else:
        return X_new, None, idx_keep


# ===================== Video-wise Train/Val Split =====================
def split_by_video(X, M, V, y, val_ratio=0.2, seed=42):
    unique_vids = np.unique(V)
    train_vids, val_vids = train_test_split(
        unique_vids, test_size=val_ratio, random_state=seed, shuffle=True
    )
    is_train = np.isin(V, train_vids)
    is_val = np.isin(V, val_vids)
    return (X[is_train], M[is_train], y[is_train]), (X[is_val], M[is_val], y[is_val]), (V[is_train], V[is_val])

# ===================== Model Definition =====================
# Masked Global Average Pooling
@register_keras_serializable()
def masked_mean(inputs):
    features, mask = inputs
    mask = tf.cast(mask, tf.float32)  # [batch, 150]
    mask = tf.expand_dims(mask, axis=-1)  # [batch, 150, 1]
    features = features * mask
    summed = tf.reduce_sum(features, axis=1)
    denom = tf.reduce_sum(mask, axis=1) + 1e-8
    return summed / denom  # [batch, features]

def TCNBlock(x, filters, kernel_size, dilation_rate, dropout, use_batch_norm=True):
    # 1D dilated causal conv block with skip connection (residual)
    prev = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation=None)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # Project input to match filters if needed for residual
    if prev.shape[-1] != filters:
        prev = tf.keras.layers.Conv1D(filters, 1, padding="same")(prev)
    x = tf.keras.layers.Add()([x, prev])
    return x

def build_rppg_model(
    window_size=150, n_channels=5, mask_as_input=True,
    conv_filters=[16, 32, 64], kernel_sizes=[1, 5, 7],
    dropout=0.2, dense_units=64, tcn_dilations=[1, 2, 4, 8],
    learning_rate=1e-3
):
    # Inputs: signals [batch, 5, 150], masks [batch, 150]
    signals_input = tf.keras.layers.Input(shape=(n_channels, window_size), name='signals')
    mask_input = tf.keras.layers.Input(shape=(window_size,), name='mask')
    
    # [batch, 5, 150] --> [batch, 150, 5] for Conv1D over time
    x = tf.keras.layers.Permute((2, 1))(signals_input)  # [batch, 150, 5]

    # Initial Conv1D block (mix channels)
    x = tf.keras.layers.Conv1D(conv_filters[0], kernel_size=kernel_sizes[0], activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(conv_filters[1], kernel_size=kernel_sizes[1], activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # TCN stack (dilated residual blocks)
    for dilation in tcn_dilations:
        x = TCNBlock(
            x,
            filters=conv_filters[2],
            kernel_size=kernel_sizes[2],
            dilation_rate=dilation,
            dropout=dropout,
            use_batch_norm=True
        )
    pooled = tf.keras.layers.Lambda(masked_mean)([x, mask_input])  # [batch, channels]
    # Dense layers
    x = tf.keras.layers.Dense(dense_units, activation='relu')(pooled)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[signals_input, mask_input], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")]
    )
    return model


# ===================== Training Script =====================
def main(args):
    seed_everything(args.seed)
    print("\nLoading data...")
    X, M, V, y = load_npz_batches(args.data_dir)  # [N, 5, 150], [N, 150], [N,], [N,]
    print(f"Loaded {X.shape[0]} samples from {len(np.unique(V))} videos.")
    # Train/val split (video-level, no window leakage)
    (X_train, M_train, y_train), (X_val, M_val, y_val), (V_train, V_val) = split_by_video(
        X, M, V, y, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"Train: {X_train.shape[0]} windows from {len(np.unique(V_train))} videos.")
    print(f"Val: {X_val.shape[0]} windows from {len(np.unique(V_val))} videos.")

    print("\nNormalizing and filtering train set...")
    X_train, M_train, idx_keep = robust_normalize(X_train, M_train, abs_max=20.0)
    y_train = y_train[idx_keep]
    print(f"Kept {X_train.shape[0]} windows after filtering (train set).")

    print("\nNormalizing and filtering val set...")
    X_val, M_val, idx_keep_val = robust_normalize(X_val, M_val, abs_max=20.0)
    y_val = y_val[idx_keep_val]
    print(f"Kept {X_val.shape[0]} windows after filtering (val set).")

    print("\nTrain label distribution:", np.bincount(y_train.astype(int)))
    print("Val label distribution:", np.bincount(y_val.astype(int)))

    checkpoint_path = os.path.join(args.out_dir, 'best_model.keras')
    initial_epoch = 19 # ===== CHANGEABLE =====
    if args.eval_only:
        print("\nEVAL ONLY MODE: Loading model and evaluating on validation set...")
        model = tf.keras.models.load_model(checkpoint_path)
        do_evaluation(model, X_val, M_val, y_val, args)
        return
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        print("Starting fresh training (no checkpoint found).")
        model = build_rppg_model(
            window_size=X.shape[2], n_channels=X.shape[1], 
            conv_filters=args.conv_filters, kernel_sizes=args.kernel_sizes,
            dropout=args.dropout, dense_units=args.dense_units, tcn_dilations=args.tcn_dilations,
            learning_rate=args.lr
        )

    print("Finished model building...")
    model.summary()
    os.makedirs(args.out_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, 'best_model.keras'),
            monitor='val_auc', save_best_only=True, mode='max', save_weights_only=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', factor=0.5, patience=5, verbose=1, mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1
        )
    ]
    # Training
    print("\nStarting model training...")
    history = model.fit(
        [X_train, M_train], y_train,
        epochs=args.epochs, batch_size=args.batch_size,
        validation_data=([X_val, M_val], y_val),
        callbacks=callbacks, shuffle=True,
        initial_epoch=initial_epoch
    )
    print("Finished model training...")

# ================== Evaluation (on validation set) ==================
def do_evaluation(model, X_val, M_val, y_val, args):
    print("Evaluating on validation set...")
    y_pred_proba = model.predict([X_val, M_val], batch_size=args.batch_size)
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    print(f"Best threshold by Youden's J: {best_thresh:.4f}")
    y_pred = (y_pred_proba >= best_thresh).astype(int)
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation AUC: {auc:.4f}, Accuracy (best threshold): {acc:.4f}")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    # Save results
    np.savez(os.path.join(args.out_dir, "eval_results.npz"),
             auc=auc, acc=acc, best_threshold=best_thresh,
             y_true=y_val, y_pred_proba=y_pred_proba, y_pred=y_pred)
    with open(os.path.join(args.out_dir, "eval_report.txt"), "w") as f:
        f.write(f"Best threshold (Youden's J): {best_thresh:.4f}\n")
        f.write(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}\n\n")
        f.write(classification_report(y_val, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_val, y_pred)))
    print("Evaluation and model checkpoint saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Directory of .npz window batches", default="/mnt/c/model_training/physio_DL_no_roi")
    parser.add_argument('--out_dir', type=str, help="Directory for output, models, metrics", default="model_training_stuff/physio_model_deeplearning/deep_learning_version_2/model")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dense_units', type=int, default=128)
    parser.add_argument('--conv_filters', nargs=3, type=int, default=[32,64,128]) # 16. 32. 64
    parser.add_argument('--kernel_sizes', nargs=3, type=int, default=[1,7,11])
    parser.add_argument('--tcn_dilations', nargs='+', type=int, default=[1,2,4,8])
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate saved model, do not train')
    args = parser.parse_args()

    main(args)
