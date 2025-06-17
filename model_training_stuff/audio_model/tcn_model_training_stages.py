# --- Script 2: Training the model ---
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
TRAIN_CACHE_DIR = "D:/model_training/cache/batches/train"
VAL_CACHE_DIR = "D:/model_training/cache/batches/val"
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

# --- TCN Residual Block ---
def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    prev_x = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
    if prev_x.shape[-1] != x.shape[-1]:
        prev_x = tf.keras.layers.Conv1D(filters, 1, padding='same')(prev_x)
    x = tf.keras.layers.Add()([x, prev_x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

    def build_model():
        inputs = tf.keras.layers.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.SpatialDropout2D(0.2)(x)
        x = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)
        for dilation_rate in [1, 2, 4, 8]:
            x = residual_block(x, filters=128, kernel_size=3, dilation_rate=dilation_rate, dropout_rate=0.2)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.models.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss=CustomFocalLoss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model

    best_val_loss = float('inf')
    reduce_lr_wait = 0
    early_stop_wait = 0
    initial_epoch = 0

    if os.path.exists(MODEL_OUTPUT_PATH):
        model = tf.keras.models.load_model(MODEL_OUTPUT_PATH, custom_objects={'CustomFocalLoss': CustomFocalLoss})
        print("Full model (with optimizer) loaded from disk.")
    else:
        model = build_model()
        print("New model built from scratch.")

    if os.path.exists(CALLBACK_STATE_PATH):
        with open(CALLBACK_STATE_PATH, 'r') as f:
            state = json.load(f)
            best_val_loss = state.get('best_val_loss', best_val_loss)
            reduce_lr_wait = state.get('reduce_lr_wait', 0)
            early_stop_wait = state.get('early_stop_wait', 0)
            initial_epoch = state.get('last_epoch', 0)

    model.summary()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    reduce_lr.wait = reduce_lr_wait
    reduce_lr.best = best_val_loss
    early_stop.wait = early_stop_wait
    early_stop.best = best_val_loss

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, save_best_only=False, verbose=1)
    # class_weights = {0: 1.5, 1: 1.0}

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=val_steps,
        epochs=initial_epoch + EPOCHS_PER_STAGE,
        initial_epoch=initial_epoch,
        callbacks=[early_stop, reduce_lr, checkpoint_cb]
    )

    model.save(MODEL_OUTPUT_PATH, include_optimizer=True)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

    val_loss_history = history.history.get('val_loss', [])
    best_val_loss = min(val_loss_history) if val_loss_history else best_val_loss
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    best_val_auc = max(history.history.get('val_auc', [0]))

    with open(METRICS_OUTPUT_PATH, 'w') as f:
        json.dump({
            'final_learning_rate': float(tf.keras.backend.get_value(model.optimizer.lr)),
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_acc,
            'best_val_auc': best_val_auc
        }, f, indent=4)
    print(f"Metrics saved to {METRICS_OUTPUT_PATH}")

    with open(CALLBACK_STATE_PATH, 'w') as f:
        json.dump({
            'reduce_lr_wait': reduce_lr.wait,
            'early_stop_wait': early_stop.wait,
            'best_val_loss': best_val_loss,
            'last_epoch': initial_epoch + len(history.history.get('loss', []))
        }, f, indent=4)
    print(f"Callback state saved to {CALLBACK_STATE_PATH}")
