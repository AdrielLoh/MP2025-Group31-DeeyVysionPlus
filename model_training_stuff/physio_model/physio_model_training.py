import os
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm

# --- Config ---
TRAIN_CACHE = 'G:/deepfake_training_datasets/faceforensics/cache/batches/train'  # Change as needed
VAL_CACHE = 'G:/deepfake_training_datasets/faceforensics/cache/batches/val'      # Change as needed
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
PATIENCE = 6

# --- Data Generator ---
def batch_generator(cache_dir, prefix, shuffle=True):
    X_files = sorted(glob(os.path.join(cache_dir, f'{prefix}_Xg_batch_*.npy')))
    y_files = sorted(glob(os.path.join(cache_dir, f'{prefix}_y_batch_*.npy')))
    indices = np.arange(len(X_files))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            X = np.load(X_files[idx])  # shape: (batch_size, window_size)
            y = np.load(y_files[idx])  # shape: (batch_size,)
            yield X, y

# --- Model Architecture: Temporal Convolutional Network (TCN) ---
def build_tcn_model(window_size):
    # TCN block using Conv1D + residuals
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
def get_window_size(cache_dir, prefix):
    X_files = sorted(glob(os.path.join(cache_dir, f'{prefix}_Xg_batch_*.npy')))
    if not X_files:
        raise RuntimeError(f'No batches found in {cache_dir} with prefix {prefix}!')
    sample = np.load(X_files[0])
    return sample.shape[1]

# --- Training script ---
def train():
    window_size = get_window_size(TRAIN_CACHE, 'real')
    print(f"Window size: {window_size}")
    model = build_tcn_model(window_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    # Count batches
    n_train_batches = len(glob(os.path.join(TRAIN_CACHE, 'real_Xg_batch_*.npy'))) + len(glob(os.path.join(TRAIN_CACHE, 'fake_Xg_batch_*.npy')))
    n_val_batches = len(glob(os.path.join(VAL_CACHE, 'real_Xg_batch_*.npy'))) + len(glob(os.path.join(VAL_CACHE, 'fake_Xg_batch_*.npy')))
    print(f"Train batches: {n_train_batches}, Val batches: {n_val_batches}")
    # Generators
    train_gen = tf.data.Dataset.from_generator(
        lambda: batch_generator(TRAIN_CACHE, 'real', shuffle=True),
        output_signature=(tf.TensorSpec(shape=(None, window_size), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int64))
    ).concatenate(
        tf.data.Dataset.from_generator(
            lambda: batch_generator(TRAIN_CACHE, 'fake', shuffle=True),
            output_signature=(tf.TensorSpec(shape=(None, window_size), tf.float32), tf.TensorSpec(shape=(None,), tf.int64))
        )
    ).unbatch().shuffle(2000).batch(BATCH_SIZE).prefetch(1)
    val_gen = tf.data.Dataset.from_generator(
        lambda: batch_generator(VAL_CACHE, 'real', shuffle=False),
        output_signature=(tf.TensorSpec(shape=(None, window_size), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int64))
    ).concatenate(
        tf.data.Dataset.from_generator(
            lambda: batch_generator(VAL_CACHE, 'fake', shuffle=False),
            output_signature=(tf.TensorSpec(shape=(None, window_size), tf.float32), tf.TensorSpec(shape=(None,), tf.int64))
        )
    ).unbatch().batch(BATCH_SIZE).prefetch(1)
    # Add channel dim for Conv1D
    def add_channel(X, y):
        return tf.expand_dims(X, -1), y
    train_gen = train_gen.map(add_channel, num_parallel_calls=tf.data.AUTOTUNE)
    val_gen = val_gen.map(add_channel, num_parallel_calls=tf.data.AUTOTUNE)
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=PATIENCE, restore_best_weights=True)
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_tcn_model.h5', monitor='val_auc', mode='max', save_best_only=True)
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=n_train_batches,
        validation_steps=n_val_batches,
        callbacks=[early_stop, checkpoint]
    )
    # Save final model
    model.save('final_tcn_model.h5')
    print("Training complete. Best model saved as best_tcn_model.h5")

if __name__ == '__main__':
    train()
