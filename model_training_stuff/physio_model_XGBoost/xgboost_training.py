import os
import numpy as np
import glob
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils import compute_sample_weight, compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import optuna

np.random.seed(42)

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

 # ===== Batch loading and stratified splitting =====
# train_dir_real = 'C:/model_training/physio_ml/real'
# train_dir_fake = 'C:/model_training/physio_ml/fake'

# print("[INFO] Loading batches...")
# X_real, y_real = load_batches(train_dir_real)
# X_fake, y_fake = load_batches(train_dir_fake)
# X_all = np.concatenate([X_real, X_fake], axis=0)
# y_all = np.concatenate([y_real, y_fake], axis=0)

# assert X_all.shape[1] == 110, f"Expected 110 features, got {X_all.shape[1]}"
# print(f"[INFO] Total samples: {X_all.shape[0]}")

# X_train, X_val, y_train, y_val = train_test_split(
#     X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
# )
# print(f"[INFO] Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
# print(f"Train class distribution: {np.bincount(y_train.astype(int))}")
# print(f"Val class distribution:   {np.bincount(y_val.astype(int))}")

# --- User: Set your directories here ---
# ===== EDIT SUCH THAT TRAIN / VAL SPLIT IS DONE AUTOMATICALLY USING SCIKIT =====
train_dir_real = 'C:/model_training/physio_ml/real'
train_dir_fake = 'C:/model_training/physio_ml/fake/validation'
val_dir_real = 'C:/model_training/physio_ml/real/validation'
val_dir_fake = 'C:/model_training/physio_ml/fake/validation'

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

# ===== Optuna Hyperparameter tuning =====
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'scale_pos_weight': scale_pos_weight,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
    }
    clf = XGBClassifier(**params)
    clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    y_score = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_score)
    return auc

print("[INFO] Running Optuna hyperparameter search...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40, show_progress_bar=True)
print("[INFO] Best hyperparameters found:")
print(study.best_params)

# --- Train XGBoost Classifier with Best Parameters ---
print("[INFO] Training XGBoost with best parameters...")
best_params = study.best_params
best_params.update({
    'scale_pos_weight': scale_pos_weight,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
})
clf = XGBClassifier(**best_params)
clf.fit(X_train, y_train, sample_weight=sample_weights)

# --- Standard Validation ---
y_pred = clf.predict(X_val)
print("\n[RESULTS] Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\n[RESULTS] Classification Report:\n", classification_report(y_val, y_pred))
print("\n[RESULTS] Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# --- ROC Curve & AUC ---
if hasattr(clf, 'predict_proba'):
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
if hasattr(clf, 'feature_importances_'):
    print("\n[INFO] Feature Importances:")
    for i, imp in enumerate(clf.feature_importances_):
        print(f"Feature {i}: {imp:.4f}")

# --- Save Model ---
joblib.dump(clf, 'models/physio_detection_xgboost_best.pkl')
print("[INFO] Model saved as models/physio_detection_xgboost_best.pkl")
