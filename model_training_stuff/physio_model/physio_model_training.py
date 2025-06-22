# --- Updated TCN Training Script with Fixes for Metric & Dataset Issues ---

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from collections import Counter

# --- Config ---
BASE_CACHE = 'D:/model_training/cache/batches'
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
PATIENCE = 6

# --- Data Generator ---
def batch_generator(cache_dir):
    X_files = sorted(glob(os.path.join(cache_dir, '*_Xg_batch_*.npy')))
    y_files = sorted(glob(os.path.join(cache_dir, '*_y_batch_*.npy')))
    for Xf, yf in zip(X_files, y_files):
        X = np.load(Xf).astype(np.float32)  # Ensure type
        y = np.load(yf).astype(np.float32)  # Ensure type
        yield X, y

# --- Model Architecture: Temporal Convolutional Network (TCN) ---
def build_tcn_model(window_size):
    inputs = tf.keras.Input(shape=(window_size, 1), name='green_signal')
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu', dilation_rate=1)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu', dilation_rate=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu', dilation_rate=4)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

# --- Discover window size from cache ---
def get_window_size(cache_dir):
    X_files = sorted(glob(os.path.join(cache_dir, '*_Xg_batch_*.npy')))
    if not X_files:
        raise RuntimeError(f'No batches found in {cache_dir}!')
    sample = np.load(X_files[0])
    return sample.shape[1]

# --- Dataset Wrapper ---
def create_dataset(cache_dir):
    ds = tf.data.Dataset.from_generator(
        lambda: batch_generator(cache_dir),
        output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    ds = ds.unbatch()
    ds = ds.map(lambda X, y: (tf.expand_dims(X, -1), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds

# --- Compute Class Weights ---
def compute_class_weights(train_dirs):
    labels = []
    for d in train_dirs:
        for y_file in sorted(glob(os.path.join(d, '*_y_batch_*.npy'))):
            y = np.load(y_file)
            labels.extend(y.tolist())
    counter = Counter(labels)
    total = sum(counter.values())
    class_weights = {cls: total / (len(counter) * count) for cls, count in counter.items()}
    print(f"Class distribution: {counter}")
    print(f"Class weights: {class_weights}")
    return class_weights

# --- Training History Plot ---
def plot_history(history):
    plt.figure(figsize=(10, 6))
    for key in history.history:
        plt.plot(history.history[key], label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# --- Training Script ---
def train():
    train_dirs = [
        os.path.join(BASE_CACHE, 'train', 'real'),
        os.path.join(BASE_CACHE, 'train', 'fake')
    ]
    val_dirs = [
        os.path.join(BASE_CACHE, 'val', 'real'),
        os.path.join(BASE_CACHE, 'val', 'fake')
    ]
    window_size = get_window_size(train_dirs[0])
    print(f"Window size: {window_size}")

    model = build_tcn_model(window_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Merge and shuffle training data across classes before batching
    train_ds = create_dataset(train_dirs[0]).concatenate(create_dataset(train_dirs[1]))
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = create_dataset(val_dirs[0]).concatenate(create_dataset(val_dirs[1]))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    class_weights = compute_class_weights(train_dirs)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=PATIENCE, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_tcn_model.h5', monitor='val_auc', mode='max', save_best_only=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weights
    )

    model.save('final_tcn_model.h5')
    print("Training complete. Best model saved as best_tcn_model.h5")

    plot_history(history)
    print("Training history saved to training_history.png")

if __name__ == '__main__':
    train()
