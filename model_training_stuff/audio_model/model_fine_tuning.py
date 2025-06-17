# --- Fine-Tuning Script: Continue Training with Lower LR & Class Weights ---
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

CHECKPOINT_DIR = "models/checkpoint"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cp.ckpt")
MODEL_OUTPUT_PATH = "models/audio_model.keras"
METRICS_OUTPUT_PATH = "models/metrics.json"
CALLBACK_STATE_PATH = "models/callback_state.json"
TRAIN_CACHE_DIR = "F:/MP-Training-Datasets/audio-bonafide-auged-more/batches/train"
VAL_CACHE_DIR = "F:/MP-Training-Datasets/audio-bonafide-auged-more/batches/val"
INPUT_HEIGHT = 170
INPUT_WIDTH = 157
INPUT_CHANNELS = 1
EPOCHS_PER_STAGE = 3

# --- Generate batches from cached data ---
def generate_cached_batches(cache_dir):
    batch_files = sorted(f for f in os.listdir(cache_dir) if f.startswith("X_batch"))
    while True:
        for f in batch_files:
            X = np.load(os.path.join(cache_dir, f))
            y = np.load(os.path.join(cache_dir, f.replace("X_batch", "y_batch")))
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            assert X.shape[0] == y.shape[0], f"Mismatch: X {X.shape} vs y {y.shape} in batch {f}"
            yield X, y

#### Focal Loss Compatible with TensorFlow 2.17.0 ####
class CustomFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        loss = alpha_t * tf.pow(1 - p_t, self.gamma) * bce
        return tf.reduce_mean(loss)

if __name__ == "__main__":
    train_generator = generate_cached_batches(TRAIN_CACHE_DIR)
    val_generator = generate_cached_batches(VAL_CACHE_DIR)
    steps_per_epoch = len([f for f in os.listdir(TRAIN_CACHE_DIR) if f.startswith("X_batch")])
    val_steps = len([f for f in os.listdir(VAL_CACHE_DIR) if f.startswith("X_batch")])

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Load existing model for fine-tuning
    if os.path.exists(MODEL_OUTPUT_PATH):
        model = tf.keras.models.load_model(MODEL_OUTPUT_PATH, custom_objects={'CustomFocalLoss': CustomFocalLoss})
        print("Model loaded from disk for fine-tuning.")
    else:
        raise FileNotFoundError("Trained model not found. Fine-tuning requires a pre-trained model.")

    # Recompile with lower LR for fine-tuning
    fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(
        optimizer=fine_tune_optimizer,
        loss=CustomFocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    #### Callbacks
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Class weights to boost real class (0)
    class_weight = {0: 1.1, 1: 1.0}

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=val_steps,
        epochs=EPOCHS_PER_STAGE,
        class_weight=class_weight,
        callbacks=[early_stop]
    )

    # Save model
    model.save(MODEL_OUTPUT_PATH, include_optimizer=True)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")
