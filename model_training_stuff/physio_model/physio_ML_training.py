import os
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt

# --- Helper: Load batches and concatenate ---
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
        return np.empty((0, 9)), np.empty((0,))

# --- User: Set your directories here ---
train_dir_real = 'D:/model_training/cache/batches/train/real'
train_dir_fake = 'D:/model_training/cache/batches/train/fake'
val_dir_real = 'D:/model_training/cache/batches/val/real'
val_dir_fake = 'D:/model_training/cache/batches/val/fake'

# --- Load all batches ---
print("[INFO] Loading batches...")
X_real_train, y_real_train = load_batches(train_dir_real)
X_fake_train, y_fake_train = load_batches(train_dir_fake)
X_real_val, y_real_val = load_batches(val_dir_real)
X_fake_val, y_fake_val = load_batches(val_dir_fake)

X_train = np.concatenate([X_real_train, X_fake_train], axis=0)
y_train = np.concatenate([y_real_train, y_fake_train], axis=0)
X_val = np.concatenate([X_real_val, X_fake_val], axis=0)
y_val = np.concatenate([y_real_val, y_fake_val], axis=0)

print(f"[INFO] Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# --- Train Random Forest ---
print("[INFO] Training Random Forest classifier...")
clf = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# --- Validation ---
y_pred = clf.predict(X_val)
print("\n[RESULTS] Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\n[RESULTS] Classification Report:\n", classification_report(y_val, y_pred))
print("\n[RESULTS] Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# --- ROC Curve & AUC ---
if hasattr(clf, 'predict_proba'):
    y_score = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_score)
    print(f"[RESULTS] ROC AUC: {auc:.4f}")
    fpr, tpr, _ = roc_curve(y_val, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Validation')
    plt.legend(loc='lower right')
    plt.show()

# --- Feature Importances ---
if hasattr(clf, 'feature_importances_'):
    print("\n[INFO] Feature Importances:")
    for i, imp in enumerate(clf.feature_importances_):
        print(f"Feature {i}: {imp:.4f}")

# --- Save Model ---
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/deepfake_detection_randomforest.pkl')
print("[INFO] Model saved as models/deepfake_detection_randomforest.pkl")
