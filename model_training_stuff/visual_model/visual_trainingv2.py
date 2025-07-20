import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import glob
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Paths
PREPROCESSED_DIR = "/mnt/d/preprocessed_data"
output_model_dir = "./model_outputs"
os.makedirs(output_model_dir, exist_ok=True)

BATCH_SIZE = 64

# --- List all batch files, split train/val by file ---
all_files = sorted(glob.glob(os.path.join(PREPROCESSED_DIR, "*.npz")))
np.random.seed(42)
np.random.shuffle(all_files)
split_idx = int(0.8 * len(all_files))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

def npz_file_generator(npz_file_list):
    for fname in npz_file_list:
        data = np.load(fname)
        feats = data["features"]
        labs = data["labels"]
        data.close()
        for i in range(len(feats)):
            yield feats[i], labs[i]

def make_tf_dataset(npz_file_list, batch_size=64, shuffle_buffer=10000, repeat=True):
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.uint8)
    )

    # Inner generator yields (feature, label) tuples
    ds = tf.data.Dataset.from_generator(
        lambda: npz_file_generator(npz_file_list),
        output_signature=output_signature
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Peek at feature shape to build model
tmp = np.load(train_files[0])
feat_dim = tmp["features"].shape[1]
tmp.close()

train_ds = make_tf_dataset(train_files, batch_size=BATCH_SIZE, shuffle_buffer=10000, repeat=True)
val_ds = make_tf_dataset(val_files, batch_size=BATCH_SIZE, shuffle_buffer=10000, repeat=False)

# Calculate steps per epoch (samples per epoch = all frames in train_files)
def count_samples(npz_file_list):
    return sum(np.load(f)["features"].shape[0] for f in npz_file_list)

train_steps = count_samples(train_files) // BATCH_SIZE
val_steps = count_samples(val_files) // BATCH_SIZE

print(f"Training on {train_steps * BATCH_SIZE} samples, validating on {val_steps * BATCH_SIZE} samples.")

# --- Model definition (same as before) ---
inputs = layers.Input(shape=(feat_dim,), dtype=tf.float32)
norm = layers.LayerNormalization(axis=1)(inputs)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L2(1e-4))(norm)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(1e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=METRICS
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
best_model_path = os.path.join(output_model_dir, "best_model.h5")
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, save_weights_only=False)

# --- Train with tf.data.Dataset streaming from disk ---
history = model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    epochs=50,
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint]
)

# Evaluate and print all metrics (on val set)
results = model.evaluate(val_ds, steps=val_steps, verbose=0)
metrics_names = model.metrics_names
for name, value in zip(metrics_names, results):
    print(f"Final model validation {name}: {value:.4f}")

# Compute F1-score 
from sklearn.metrics import f1_score, confusion_matrix

all_y_true, all_y_pred = [], []
for x_batch, y_batch in val_ds.take(val_steps):
    preds = (model.predict(x_batch, verbose=0) > 0.5).astype(np.uint8).flatten()
    all_y_true.extend(y_batch.numpy())
    all_y_pred.extend(preds)

f1 = f1_score(all_y_true, all_y_pred)
cm = confusion_matrix(all_y_true, all_y_pred)
print(f"Final model validation F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# --- Save weights in HDF5 format (recommended for TF/Keras) ---
final_weights_path = os.path.join(output_model_dir, "final_weights.weights.h5")
model.save_weights(final_weights_path)  # Save weights of last (final) model

best_weights_path = os.path.join(output_model_dir, "best_weights.weights.h5")
best_model = tf.keras.models.load_model(best_model_path)  # Load best model from checkpoint
best_model.save_weights(best_weights_path)  # Save best weights only

print(f"Saved final weights to: {final_weights_path}")
print(f"Saved best weights to: {best_weights_path}")
