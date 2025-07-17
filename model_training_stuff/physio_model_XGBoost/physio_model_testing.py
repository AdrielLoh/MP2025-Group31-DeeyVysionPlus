import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
MODEL_PATH = 'models/physio_detection_xgboost_best.pkl'
SCALER_PATH = 'models/physio_scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Helper: Load test batches
def load_batches(batch_dir):
    X, y = [], []
    x_files = sorted(glob.glob(os.path.join(batch_dir, '*_Xrppg_batch_*.npy')))
    for x_file in x_files:
        y_file = x_file.replace('_Xrppg_', '_y_')
        X.append(np.load(x_file))
        y.append(np.load(y_file))
    if X:
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    else:
        return np.empty((0, 57)), np.empty((0,))

# Set your test dirs here
# Example: Replace with your actual paths

test_dir_real = 'D:/model_training/cache/batches/test/real'
test_dir_fake = 'D:/model_training/cache/batches/test/fake'

print("[INFO] Loading test batches...")
X_real, y_real = load_batches(test_dir_real)
X_fake, y_fake = load_batches(test_dir_fake)
X_test = np.concatenate([X_real, X_fake], axis=0)
y_test = np.concatenate([y_real, y_fake], axis=0)

print(f"[INFO] Test samples: {X_test.shape[0]}")

print("[INFO] Normalizing features with StandardScaler...")
X_test = scaler.transform(X_test)

print("[INFO] Running model on test set...")
y_pred = model.predict(X_test)

print("\n[RESULTS] Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n[RESULTS] Classification Report:\n", classification_report(y_test, y_pred))
print("\n[RESULTS] Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if hasattr(model, 'predict_proba'):
    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    print(f"[RESULTS] ROC AUC: {auc:.4f}")
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test')
    plt.legend(loc='lower right')
    plt.show()

    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"[INFO] Optimal Threshold by Youden's J Index: {optimal_threshold:.4f}")

    y_pred_optimal = (y_score >= optimal_threshold).astype(int)
    print("\n[Optimal Threshold] Classification Report:\n", classification_report(y_test, y_pred_optimal))
    print("\n[Optimal Threshold] Confusion Matrix:\n", confusion_matrix(y_test, y_pred_optimal))
