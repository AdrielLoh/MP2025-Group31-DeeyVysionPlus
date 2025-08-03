import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
# GPU memory growth
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

# listing all npz files and split them into training and validation splits
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

#using tensorflow data generator to feed the model with data without overwhelming the vram and ram
def make_tf_dataset(npz_file_list, batch_size=64, shuffle_buffer=10000, repeat=True):
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.uint8)
    )
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

# Calculate steps per epoch
def count_samples(npz_file_list):
    return sum(np.load(f)["features"].shape[0] for f in npz_file_list)

train_steps = count_samples(train_files) // BATCH_SIZE
val_steps = count_samples(val_files) // BATCH_SIZE

print(f"Training on {train_steps * BATCH_SIZE} samples, validating on {val_steps * BATCH_SIZE} samples.")

# model architecture code with dropouts, regularization 
inputs = layers.Input(shape=(feat_dim,), dtype=tf.float32)
x = layers.LayerNormalization(axis=1)(inputs)
x = layers.Dense(512, kernel_regularizer=regularizers.L2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(384, kernel_regularizer=regularizers.L2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, kernel_regularizer=regularizers.L2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, kernel_regularizer=regularizers.L2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, kernel_regularizer=regularizers.L2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.2)(x)
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),  # Label smoothing for robustness
    metrics=METRICS
)

#early stopping mechanism if the val_loss value does not improve for 7 epochs, preventing overfitting of the model
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
best_model_path = os.path.join(output_model_dir, "best_model.h5")
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, save_weights_only=False)

history = model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    epochs=60,
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint]
)

# calculating training metrics based on validation set
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

# save model weights in HDF5 format 
final_weights_path = os.path.join(output_model_dir, "final_weights.weights.h5")
model.save_weights(final_weights_path)  # save weights of last (final) model after completing 60 epochs

best_weights_path = os.path.join(output_model_dir, "best_weights.weights.h5")
best_model = tf.keras.models.load_model(best_model_path)
best_model.save_weights(best_weights_path)  # save best weights with the lowest val_loss value

print(f"Saved final weights to: {final_weights_path}")
print(f"Saved best weights to: {best_weights_path}")

#training metric graphs generation
all_y_true, all_y_pred, all_y_prob = [], [], []
for x_batch, y_batch in val_ds.take(val_steps):
    probas = model.predict(x_batch, verbose=0).flatten()
    preds = (probas > 0.5).astype(np.uint8)
    all_y_true.extend(y_batch.numpy())
    all_y_pred.extend(preds)
    all_y_prob.extend(probas)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_prob = np.array(all_y_prob)

# confusion Matrix 
plt.figure(figsize=(6, 5))
cm = confusion_matrix(all_y_true, all_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'confusion_matrix.png'))
plt.close()

# precision-recall Curve
precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
plt.figure()
plt.plot(recall, precision, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'precision_recall_curve.png'))
plt.close()

# prediction probability distribution 
plt.figure()
plt.hist(all_y_prob, bins=40, color='skyblue', alpha=0.8)
plt.xlabel('Predicted Probability of Being Fake')
plt.ylabel('Sample Count')
plt.title('Prediction Probability Distribution')
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'prob_distribution.png'))
plt.close()

# prediction probability distribution by class
plt.figure()
plt.hist([all_y_prob[all_y_true == 0], all_y_prob[all_y_true == 1]], bins=40,
         label=['Real', 'Fake'], color=['green', 'red'], alpha=0.7, stacked=True)
plt.xlabel('Predicted Probability of Being Fake')
plt.ylabel('Sample Count')
plt.title('Prediction Probability Distribution by Class')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'prob_dist_by_class.png'))
plt.close()

# ROC curve 
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'roc_curve.png'))
plt.close()