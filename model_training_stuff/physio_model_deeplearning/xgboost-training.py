import os
import glob
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import argparse
import logging
import json
import random
import gc

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("physio_xgb")

# ------------------------------------
# Arguments and Config
# ------------------------------------
def get_config():
    parser = argparse.ArgumentParser(description='Train XGBoost model for physio deepfake detection')
    parser.add_argument('--data_dir', type=str, default='D:/model_training/cache/batches/physio-deep-v1/for-training', help='Directory with preprocessed HDF5 files')
    parser.add_argument('--model_dir', type=str, default='D:/model_training/physiological-model/xgb-ml-1', help='Directory to save trained models')
    parser.add_argument('--log_dir', type=str, default='D:/model_training/physiological-model/xgb-ml-1/logs', help='Directory to save logs')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    return args

# ------------------------------------
# Data Utilities
# ------------------------------------

def extract_label_from_filename(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'dataset_label' in f.attrs:
                label_str = f.attrs['dataset_label']
                if isinstance(label_str, (bytes, np.bytes_)):
                    label_str = label_str.decode('utf-8')
                return 1 if 'fake' in str(label_str).lower() else 0
    except Exception as e:
        logger.warning(f"Could not read label from {filepath}: {e}")
    return 0

def build_ml_window_metadata(hdf5_files):
    """Collect window-level ML features for all samples."""
    rows = []
    for h5_path in tqdm(hdf5_files, desc="Indexing HDF5 files"):
        label = extract_label_from_filename(h5_path)
        try:
            with h5py.File(h5_path, 'r') as f:
                for group_name in f.keys():
                    g = f[group_name]
                    video_id = g.attrs.get('original_filename', group_name)
                    if isinstance(video_id, bytes):
                        video_id = video_id.decode()
                    for win_name in g.keys():
                        if not win_name.startswith('window_'):
                            continue
                        win_group = g[win_name]
                        if 'ml_features' in win_group:
                            feats = {}
                            ml_feats_group = win_group['ml_features']
                            for k in ml_feats_group.keys():
                                feats[k] = float(ml_feats_group[k][()])
                            rows.append({
                                'file_path': h5_path,
                                'group_name': group_name,
                                'window_name': win_name,
                                'label': label,
                                'video_id': video_id,
                                **feats  # flatten all features into columns
                            })
        except Exception as e:
            logger.warning(f"Error reading {h5_path}: {e}")
    df = pd.DataFrame(rows)
    return df

def stratified_group_window_split(df, n_splits=5, random_state=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(df, df['label'], groups=df['video_id']))
    return splits  # (train_indices, val_indices)

# ------------------------------------
# XGBoost Training
# ------------------------------------
def main():
    config = get_config()
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Find HDF5 files
    all_files = sorted(glob.glob(os.path.join(config.data_dir, "*.h5")))
    if not all_files:
        logger.error(f"No HDF5 files found in {config.data_dir}")
        return

    logger.info(f"Found {len(all_files)} HDF5 files")

    # Build DataFrame with window-level ML features
    df = build_ml_window_metadata(all_files)
    logger.info(f"Loaded {len(df)} window samples with ML features")
    splits = stratified_group_window_split(df, n_splits=config.folds, random_state=config.seed)

    feature_cols = [col for col in df.columns if col not in ['file_path', 'group_name', 'window_name', 'label', 'video_id']]
    logger.info(f"Number of ML features: {len(feature_cols)}")

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"===== Fold {fold+1}/{config.folds} =====")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['label'].values

        # XGBoost params (can Optuna or manual tune later)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',  # fast, works on CPU and AMD
            'random_state': config.seed,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'n_estimators': 500,
            'verbosity': 1,
        }

        clf = xgb.XGBClassifier(**params)

        logger.info("Training XGBoost model...")
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=12,
            verbose=True
        )

        # Evaluate
        y_pred_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)
        auroc = roc_auc_score(y_val, y_pred_prob)
        acc = accuracy_score(y_val, y_pred)

        # Find best threshold
        best_acc = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.9, 81):
            acc_thresh = accuracy_score(y_val, y_pred_prob > thresh)
            if acc_thresh > best_acc:
                best_acc = acc_thresh
                best_thresh = thresh

        # Save model
        model_path = os.path.join(config.model_dir, f"xgb_fold{fold+1}.json")
        clf.save_model(model_path)
        logger.info(f"Model for fold {fold+1} saved to {model_path}")

        # Log
        result = {
            'fold': fold + 1,
            'auroc': float(auroc),
            'accuracy': float(acc),
            'best_accuracy': float(best_acc),
            'best_threshold': float(best_thresh),
            'n_train': int(len(train_df)),
            'n_val': int(len(val_df)),
            'feature_cols': feature_cols,
        }
        fold_results.append(result)
        with open(os.path.join(config.log_dir, f"xgb_fold{fold+1}_results.json"), 'w') as f:
            json.dump(result, f, indent=2)

        # Feature importance
        importance = clf.get_booster().get_score(importance_type='gain')
        importance_sorted = sorted(importance.items(), key=lambda x: -x[1])
        logger.info(f"Top 10 features for fold {fold+1}:")
        for name, score in importance_sorted[:10]:
            logger.info(f"{name}: {score:.4f}")

        gc.collect()

    # Summary
    mean_auroc = np.mean([r['auroc'] for r in fold_results])
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    mean_best_acc = np.mean([r['best_accuracy'] for r in fold_results])
    logger.info("==== XGBoost CV Summary ====")
    logger.info(f"AUROC: {mean_auroc:.4f}")
    logger.info(f"Accuracy (0.5): {mean_acc:.4f}")
    logger.info(f"Best Accuracy: {mean_best_acc:.4f}")
    with open(os.path.join(config.log_dir, "xgb_cv_results.json"), 'w') as f:
        json.dump(fold_results, f, indent=2)

if __name__ == "__main__":
    main()
