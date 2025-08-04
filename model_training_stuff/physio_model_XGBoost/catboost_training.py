import os
import numpy as np
import glob
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import compute_sample_weight, compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import optuna
import random

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
        return np.empty((0, 110)), np.empty((0,))

train_dir_real = 'C:/model_training/physio_ml/real'
train_dir_fake = 'C:/model_training/physio_ml/fake'

print("[INFO] Loading batches...")
X_real, y_real = load_batches(train_dir_real)
X_fake, y_fake = load_batches(train_dir_fake)
X_all = np.concatenate([X_real, X_fake], axis=0)
y_all = np.concatenate([y_real, y_fake], axis=0)

assert X_all.shape[1] == 110, f"Expected 110 features, got {X_all.shape[1]}"
print(f"[INFO] Total samples: {X_all.shape[0]}")

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=random.randint(1, 10000)
)
print(f"[INFO] Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
print(f"Train class distribution: {np.bincount(y_train.astype(int))}")
print(f"Val class distribution:   {np.bincount(y_val.astype(int))}")

# ===== Feature trimming to mitigate overfitting =====
X_train = X_train[:, :-4]
X_val = X_val[:, :-4]

print(f"[INFO] Feature shape after trimming: {X_train.shape}")

# ===== Fitting a standard scaler for normalization =====
print("[INFO] Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/physio_scaler.pkl')
print("[INFO] Feature scaler saved to models/physio_scaler.pkl")

# ===== Calculating class weights automatically =====
scale_pos_weight = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
print(f"Auto-calculated scale_pos_weight: {scale_pos_weight:.4f}")
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
print(f"Auto-calculated class weights: {class_weight_dict}")
sample_weights = compute_sample_weight(class_weight=class_weight_dict, y=y_train)

# ===== Optuna Hyperparameter tuning for CatBoost =====
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 500),
        "depth": trial.suggest_int("depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "random_seed": random.randint(1, 10000),
        "verbose": False,
        "task_type": "CPU",
    }
    clf = CatBoostClassifier(**params)
    clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    y_score = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_score)
    return auc

print("[INFO] Running Optuna hyperparameter search (CatBoost)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40, show_progress_bar=True)
print("[INFO] Best hyperparameters found:")
print(study.best_params)

# --- Train CatBoost Classifier with Best Parameters ---
print("[INFO] Training CatBoost with best parameters...")
best_params = study.best_params
best_params.update({
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "random_seed": random.randint(1, 10000),
    "verbose": False,
    "task_type": "CPU"
})
clf = CatBoostClassifier(**best_params)
clf.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# --- Standard Validation ---
y_pred = clf.predict(X_val)
print("\n[RESULTS] Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\n[RESULTS] Classification Report:\n", classification_report(y_val, y_pred))
print("\n[RESULTS] Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# --- ROC Curve & AUC ---
if hasattr(clf, "predict_proba"):
    y_score = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_score)
    print(f"[RESULTS] ROC AUC: {auc:.4f}")
    fpr, tpr, thresholds = roc_curve(y_val, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Validation')
    plt.legend(loc='lower right')
    plt.show()

    # --- Optimal Threshold using Youden's J Index ---
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"[INFO] Optimal Threshold by Youden's J Index: {optimal_threshold:.4f}")

    y_pred_optimal = (y_score >= optimal_threshold).astype(int)
    print("\n[Optimal Threshold] Classification Report:\n", classification_report(y_val, y_pred_optimal))
    print("\n[Optimal Threshold] Confusion Matrix:\n", confusion_matrix(y_val, y_pred_optimal))

# --- Feature Importances ---
if hasattr(clf, "get_feature_importance"):
    print("\n[INFO] Feature Importances:")
    importances = clf.get_feature_importance()
    for i, imp in enumerate(importances):
        print(f"Feature {i}: {imp:.4f}")

# --- Save Model ---
joblib.dump(clf, "models/physio_detection_catboost_best.pkl")
print("[INFO] Model saved as models/physio_detection_catboost_best.pkl")
