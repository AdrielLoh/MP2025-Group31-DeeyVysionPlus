import os
import numpy as np
import tensorflow as tf
import random
from glob import glob
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import argparse

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
                all_labels.append(np.full(d["signals"].shape[0], int(label)))
            else:
                raise RuntimeError("Each npz file must have 'label' entry")
    X = np.concatenate(all_signals, axis=0)     # [N, 5, 150]
    M = np.concatenate(all_masks, axis=0)       # [N, 150]
    V = np.concatenate(all_vids, axis=0)        # [N,]
    y = np.concatenate(all_labels, axis=0)      # [N,]
    return X, M, V, y

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
def build_rppg_model(
    window_size=150, n_channels=5, mask_as_input=True,
    conv_filters=[16, 32, 64], kernel_sizes=[1, 5, 7], 
    dropout=0.2, dense_units=64, tcn_dilations=[1, 2, 4, 8],
    learning_rate=1e-3
):
    from tcn import TCN

    # Inputs: signals [batch, 5, 150], masks [batch, 150]
    signals_input = tf.keras.layers.Input(shape=(n_channels, window_size), name='signals')
    mask_input = tf.keras.layers.Input(shape=(window_size,), name='mask')
    
    # Transpose to [batch, 150, 5] for Conv1D
    x = tf.keras.layers.Permute((2,1))(signals_input)   # now [batch, 150, 5]

    # Channel-mixing Conv1D (kernel=1)
    x = tf.keras.layers.Conv1D(conv_filters[0], kernel_size=kernel_sizes[0], activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # Temporal Conv1D layers
    x = tf.keras.layers.Conv1D(conv_filters[1], kernel_size=kernel_sizes[1], activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(conv_filters[2], kernel_size=kernel_sizes[2], activation='relu', padding='same', dilation_rate=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # Temporal Convolutional Network (TCN)
    x = TCN(nb_filters=conv_filters[2], kernel_size=kernel_sizes[2], dilations=tcn_dilations, 
            use_batch_norm=True, use_weight_norm=True, dropout_rate=dropout)(x)
    # Masked Global Average Pooling
    def masked_mean(inputs):
        features, mask = inputs
        mask = tf.cast(mask, tf.float32)  # [batch, 150]
        mask = tf.expand_dims(mask, axis=-1)  # [batch, 150, 1]
        features = features * mask  # zero out invalid
        summed = tf.reduce_sum(features, axis=1)
        denom = tf.reduce_sum(mask, axis=1) + 1e-8
        return summed / denom  # [batch, features]
    pooled = tf.keras.layers.Lambda(masked_mean)([x, mask_input]) # [batch, channels]
    # Dense layers
    x = tf.keras.layers.Dense(dense_units, activation='relu')(pooled)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[signals_input, mask_input], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")])
    return model

# ===================== Training Script =====================
def main(args):
    seed_everything(args.seed)
    print("Loading data...")
    X, M, V, y = load_npz_batches(args.data_dir)  # [N, 5, 150], [N, 150], [N,], [N,]
    print(f"Loaded {X.shape[0]} samples from {len(np.unique(V))} videos.")
    # Train/val split (video-level, no window leakage)
    (X_train, M_train, y_train), (X_val, M_val, y_val), (V_train, V_val) = split_by_video(
        X, M, V, y, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"Train: {X_train.shape[0]} windows from {len(np.unique(V_train))} videos.")
    print(f"Val: {X_val.shape[0]} windows from {len(np.unique(V_val))} videos.")

    # Model
    model = build_rppg_model(
        window_size=X.shape[2], n_channels=X.shape[1], 
        conv_filters=args.conv_filters, kernel_sizes=args.kernel_sizes,
        dropout=args.dropout, dense_units=args.dense_units, tcn_dilations=args.tcn_dilations,
        learning_rate=args.lr
    )
    model.summary()
    os.makedirs(args.out_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, 'best_model.h5'),
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
    history = model.fit(
        [X_train, M_train], y_train,
        epochs=args.epochs, batch_size=args.batch_size,
        validation_data=([X_val, M_val], y_val),
        callbacks=callbacks, shuffle=True
    )

    # ================== Evaluation (on validation set) ==================
    # Reload best model
    model = tf.keras.models.load_model(os.path.join(args.out_dir, 'best_model.h5'),
                                       custom_objects={"TCN": __import__('tcn').TCN})
    y_pred_proba = model.predict([X_val, M_val], batch_size=args.batch_size)
    # Find best threshold by Youden's J index
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

    # Save all results
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
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of .npz window batches")
    parser.add_argument('--out_dir', type=str, default='model_out', help="Directory for output, models, metrics")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dense_units', type=int, default=64)
    parser.add_argument('--conv_filters', nargs=3, type=int, default=[16,32,64])
    parser.add_argument('--kernel_sizes', nargs=3, type=int, default=[1,5,7])
    parser.add_argument('--tcn_dilations', nargs='+', type=int, default=[1,2,4,8])
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
