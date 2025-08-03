import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score
import seaborn as sns

# GPU memory growth mechanism to prevent using up all vram
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Enable memory growth for each GPU
    except RuntimeError as e:
        print(e)    

PREPROCESSED_DIR = "/mnt/d/preprocessed_data"
output_model_dir = "./model_outputs"     
os.makedirs(output_model_dir, exist_ok=True)

BATCH_SIZE = 64 # number of samples per training batch

# prep dataset splits 
all_files = sorted(glob.glob(os.path.join(PREPROCESSED_DIR, "*.npz")))  #
np.random.seed(42)
np.random.shuffle(all_files)
split_idx = int(0.8 * len(all_files))# 80% of preprocessed data is used for training
train_files = all_files[:split_idx]# splitting the first 80% of the data
val_files = all_files[split_idx:]# validation files are the remaining 20% of the split

# data generator
def npz_file_generator(npz_file_list):
    for fname in npz_file_list:
        data = np.load(fname)
        feats = data["features"]# features array extraction. each row of features has a feature vector for one sample and dimension (feat_dim)
        labs = data["labels"]# labels array extraction
        data.close()
        for i in range(len(feats)):
            yield feats[i], labs[i]# yield one (feature, label) pair at a time to save memory 

# TensorFlow Dataset tool
def make_tf_dataset(npz_file_list, batch_size=64, shuffle_buffer=10000, repeat=True):
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),# feature vectors are in float32 format
        tf.TensorSpec(shape=(), dtype=tf.uint8)# every label is one integer
    )
    ds = tf.data.Dataset.from_generator(
        lambda: npz_file_generator(npz_file_list),
        output_signature=output_signature #providing data format for tensorflow
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer)#shuffle within buffer for randomness
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)#prefetch next batches for better training efficiency
    return ds 

# check first training batch to obtain feature dimension 
tmp = np.load(train_files[0])
feat_dim = tmp["features"].shape[1]# number of features per sample
tmp.close()

# create streaming TF datasets for training and validation based on split created earlier
train_ds = make_tf_dataset(train_files, batch_size=BATCH_SIZE, shuffle_buffer=10000, repeat=True)
val_ds = make_tf_dataset(val_files, batch_size=BATCH_SIZE, shuffle_buffer=10000, repeat=False)

#count number of samples in a set of .npz files
def count_samples(npz_file_list):
    return sum(np.load(f)["features"].shape[0] for f in npz_file_list)

train_steps = count_samples(train_files) // BATCH_SIZE  # steps per epoch for training based on number of samples in training files
val_steps = count_samples(val_files) // BATCH_SIZE     

print(f"Training on {train_steps * BATCH_SIZE} samples, validating on {val_steps * BATCH_SIZE} samples.")

# model architecture
inputs = layers.Input(shape=(feat_dim,), dtype=tf.float32)# input layer for feature vectors
x = layers.LayerNormalization(axis=1)(inputs)# normalize input features
x = layers.Dense(512, kernel_regularizer=regularizers.L2(1e-4))(x)  # Dense + L2 regularization
x = layers.BatchNormalization()(x)# batch normalization for stability
x = layers.LeakyReLU(alpha=0.1)(x)# LeakyReLU activation
x = layers.Dropout(0.4)(x)#dropout for regularization
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
outputs = layers.Dense(1, activation='sigmoid')(x)# output probability (real/fake)
model = Model(inputs=inputs, outputs=outputs)# define model object
model.summary()# print architecture

# model compiling and training metrics
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),# standard accuracy
    tf.keras.metrics.Precision(name='precision'),# precision
    tf.keras.metrics.Recall(name='recall'),# recall
    tf.keras.metrics.AUC(name='auc')# ROC AUC
]
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),  # Adam optimizer to adjust learning rates
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),  # smoothing for robustness 
    metrics=METRICS
)

# early stopping and model checkpointing 
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True) # stop training if no val_loss improvement for 7 epochs
best_model_path = os.path.join(output_model_dir, "best_model.h5")
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, save_weights_only=False)

# model training
history = model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    epochs=60,
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint]
)

# evaluate model on validation data
results = model.evaluate(val_ds, steps=val_steps, verbose=0)# returns list of metric values
metrics_names = model.metrics_names
for name, value in zip(metrics_names, results):
    print(f"Final model validation {name}: {value:.4f}")

# validation predictions, probabilities, and actual labels 
all_y_true, all_y_pred, all_y_prob = [], [], []# initialize lists
for x_batch, y_batch in val_ds.take(val_steps):# loop over all validation batches
    probas = model.predict(x_batch, verbose=0).flatten()# predicted probabilities (float [0,1]) by the model
    preds = (probas > 0.5).astype(np.uint8)# threshold
    all_y_true.extend(y_batch.numpy())# actual labels
    all_y_pred.extend(preds)# predicted class
    all_y_prob.extend(probas)# predicted probability

all_y_true = np.array(all_y_true)  # convert lists to numpy arrays (for plotting/metrics)
all_y_pred = np.array(all_y_pred)
all_y_prob = np.array(all_y_prob)

f1 = f1_score(all_y_true, all_y_pred)      # F1-score
cm = confusion_matrix(all_y_true, all_y_pred) # Confusion matrix
print(f"Final model validation F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# save model weights (final and best)
final_weights_path = os.path.join(output_model_dir, "final_weights.weights.h5")
model.save_weights(final_weights_path)                         # save weights after all epochs (may be after early stopping)
best_weights_path = os.path.join(output_model_dir, "best_weights.weights.h5")
best_model = tf.keras.models.load_model(best_model_path)       # load best model for saving weights separately
best_model.save_weights(best_weights_path)
print(f"Saved final weights to: {final_weights_path}")
print(f"Saved best weights to: {best_weights_path}")

# confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'confusion_matrix.png'))
plt.close()

#precision-recall curve
precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
plt.figure()
plt.plot(recall, precision, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'precision_recall_curve.png'))
plt.close()

#histogram of all predicted probabilities
plt.figure()
plt.hist(all_y_prob, bins=40, color='skyblue', alpha=0.8)
plt.xlabel('Predicted Probability of Being Fake')
plt.ylabel('Sample Count')
plt.title('Prediction Probability Distribution')
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'prob_distribution.png'))
plt.close()

#predicted probabilities by class (real vs fake)
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

#ROC curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)         # Calculate ROC
roc_auc = auc(fpr, tpr)                                 # Area under ROC
plt.figure()
plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')  # Diagonal (random classifier)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_model_dir, 'roc_curve.png'))
plt.close()
