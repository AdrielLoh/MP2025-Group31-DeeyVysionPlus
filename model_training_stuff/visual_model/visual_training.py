import numpy as np
import glob
import os
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def get_npz_batches(directory):
    # Accept any npz batch file
    return sorted(glob.glob(os.path.join(directory, '*_data_batch_*.npz')))

def batch_generator(files):
    while True:
        np.random.shuffle(files)
        for fname in files:
            data = np.load(fname)
            X, y = data['features'], data['labels']
            # Filter out zero-variance samples
            valid = np.logical_and(np.linalg.norm(X, axis=1) > 1e-8, np.std(X, axis=1) > 1e-8)
            X, y = X[valid], y[valid]
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            yield X[idx], y[idx]

def train_val_split(batch_files, val_ratio=0.15):
    batch_files = shuffle(batch_files, random_state=42)
    n_val = int(len(batch_files) * val_ratio)
    return batch_files[n_val:], batch_files[:n_val]

def eval_model(model, files, batch_size=1024):
    y_true, y_pred_probs = [], []
    for fname in files:
        data = np.load(fname)
        X, y = data['features'], data['labels']
        valid = np.logical_and(np.linalg.norm(X, axis=1) > 1e-8, np.std(X, axis=1) > 1e-8)
        X, y = X[valid], y[valid]
        if len(X) == 0:
            continue
        preds = model.predict(X, batch_size=batch_size, verbose=0).flatten()
        y_true.extend(y)
        y_pred_probs.extend(preds)
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)
    return cm, f1, roc_auc, pr_auc, fpr, tpr, precision, recall, y_true, y_pred_probs, y_pred_labels

def plot_training_curves(history, output_dir):
    df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df[['accuracy', 'val_accuracy']].plot(ax=axes[0])
    axes[0].set_title('Accuracy')
    df[['loss', 'val_loss']].plot(ax=axes[1])
    axes[1].set_title('Loss')
    if 'auc' in df.columns and 'val_auc' in df.columns:
        df[['auc', 'val_auc']].plot(ax=axes[2])
        axes[2].set_title('AUC')
    for ax in axes: ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def plot_roc_pr(fpr, tpr, roc_auc, precision, recall, pr_auc, output_dir):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC={pr_auc:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

def plot_confusion_matrix(cm, output_dir, classes=['Real', 'Fake']):
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/d/preprocessed_data')
    parser.add_argument('--output_dir', type=str, default='/mnt/d/trained_models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    final_model_path = os.path.join(args.output_dir, 'final_model.keras')
    best_model_path = os.path.join(args.output_dir, 'best_model.keras')
    final_weights_path = os.path.join(args.output_dir, 'final_weights.weights.h5')
    best_weights_path = os.path.join(args.output_dir, 'best_weights.weights.h5')
    log_path = os.path.join(args.output_dir, 'training_log.csv')

    # Batch files and split
    batch_files = get_npz_batches(args.data_dir)
    train_files, val_files = train_val_split(batch_files, val_ratio=args.val_ratio)

    # Get input shape from a sample
    sample = np.load(train_files[0])
    input_shape = sample['features'].shape[1:]

    # Build model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        best_model_path, save_best_only=True, monitor='val_loss'
    )
    earlystop_cb = callbacks.EarlyStopping(
        patience=args.patience, restore_best_weights=False, monitor='val_loss'
    )
    csvlogger_cb = callbacks.CSVLogger(log_path)

    train_steps = len(train_files)
    val_steps = len(val_files)

    # Training
    history = model.fit(
        batch_generator(train_files),
        steps_per_epoch=train_steps,
        validation_data=batch_generator(val_files),
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, earlystop_cb, csvlogger_cb],
        verbose=1
    )

    # Save final model/weights
    model.save(final_model_path)
    model.save_weights(final_weights_path)

    # Save best weights if available (for .weights.h5 export)
    if os.path.exists(best_model_path):
        best_model = tf.keras.models.load_model(best_model_path)
        best_model.save_weights(best_weights_path)
    else:
        best_model = None

    print('Training complete. Models and weights saved.')
    plot_training_curves(history, args.output_dir)

    # Evaluation (on val_files as "test" set here)
    print("Evaluating best model on validation set...")
    eval_model_path = best_model_path if best_model else final_model_path
    model = tf.keras.models.load_model(eval_model_path)
    cm, f1, roc_auc, pr_auc, fpr, tpr, precision, recall, y_true, y_pred_probs, y_pred_labels = eval_model(model, val_files, batch_size=args.batch_size)

    print("Confusion Matrix:\n", cm)
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")

    plot_roc_pr(fpr, tpr, roc_auc, precision, recall, pr_auc, args.output_dir)
    plot_confusion_matrix(cm, args.output_dir)

    # Save detailed results for reproducibility/auditing
    np.savez(os.path.join(args.output_dir, "eval_preds.npz"),
             y_true=y_true, y_pred_probs=y_pred_probs, y_pred_labels=y_pred_labels)
    print("All metrics and plots are saved in:", args.output_dir)
