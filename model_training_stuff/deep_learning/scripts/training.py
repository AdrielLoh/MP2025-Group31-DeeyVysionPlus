import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import optuna
from tqdm import tqdm
import logging
import gc
import argparse
import json
import pandas as pd
from datetime import datetime
import time

#Temporary parse for log_dir
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--log_dir', type=str, default='C:/deepvysion/logs')
_args, _ = _parser.parse_known_args()


# Create logger immediately
def setup_logging(log_dir):
    import os
    os.makedirs(log_dir, exist_ok=True)
    handlers = [
        logging.FileHandler(os.path.join(log_dir, 'training_progress.log')),
        logging.StreamHandler()
    ]
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=handlers)
    return logging.getLogger(__name__)

logger = setup_logging(_args.log_dir)

def log_with_timestamp(message):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}", flush=True)
    logger.info(message)

# --- GPU Setup for RTX 3060 Ti ---
def setup_rtx_3060_ti():
    log_with_timestamp("Setting up RTX 3060 Ti GPU...")
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)]
            )
            log_with_timestamp(f"Found {len(gpus)} GPU(s); limited to 7GB")
        else:
            log_with_timestamp("WARNING: No GPU found, using CPU")
        return True
    except Exception as e:
        log_with_timestamp(f"GPU setup failed: {e}")
        return False

def setup_mixed_precision():
    log_with_timestamp("Setting up mixed precision training...")
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        log_with_timestamp("Mixed precision (float16) enabled")
        return True
    except Exception as e:
        log_with_timestamp(f"Mixed precision setup failed: {e}")
        return False

# --- Configuration ---
def get_config():
    log_with_timestamp("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--data_dir',   type=str, default='C:/deepvysion/preprocessed-1')
    parser.add_argument('--model_dir',  type=str, default='C:/deepvysion/deepfake_models')
    parser.add_argument('--log_dir',    type=str, default='C:/deepvysion/logs')
    parser.add_argument('--epochs',     type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size',   type=int, default=128)
    parser.add_argument('--optuna_trials', type=int, default=10)
    parser.add_argument('--skip_optuna',   action='store_true')
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_with_timestamp(f"Model directory: {args.model_dir}")
    log_with_timestamp(f"Log directory:   {args.log_dir}")
    return args

# --- Data Generator ---
class DeepfakeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_files, batch_size, img_size, shuffle, augment, name):
        self.batch_files = batch_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.name = name
        self.indexes = np.arange(len(batch_files))
        self.on_epoch_end()
        self.batch_count = 0
        log_with_timestamp(f"{self.name} initialized: {len(batch_files)} batches")

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        start = time.time()
        self.batch_count += 1
        actual_idx = self.indexes[idx]
        batch_file = self.batch_files[actual_idx]
        log_with_timestamp(f"{self.name} batch {self.batch_count}/{len(self)} loading {os.path.basename(batch_file)}")
        frames = np.load(batch_file, allow_pickle=True)
        labels = np.load(batch_file.replace('_frames.npy','_labels.npy'), allow_pickle=True)
        X, y = [], []
        valid = 0
        for vf, lbl in zip(frames, labels):
            if vf is None or len(vf)==0: continue
            valid += 1
            sel = np.random.choice(len(vf), min(len(vf),8), replace=False)
            for i in sel:
                f = vf[i]
                if f is None or f.shape!=(self.img_size,self.img_size,3): continue
                if self.augment:
                    f = self._augment(f)
                X.append(f)
                y.append(lbl)
        if not X:
            log_with_timestamp(f"{self.name} WARNING: empty batch returned")
            return np.zeros((0,self.img_size,self.img_size,3),dtype=np.float32), np.array([])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        if len(X)>self.batch_size:
            idxs = np.random.choice(len(X), self.batch_size, replace=False)
            X, y = X[idxs], y[idxs]
        log_with_timestamp(f"{self.name} batch {self.batch_count} done in {time.time()-start:.2f}s, valid_videos={valid}, batch_shape={X.shape}")
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            log_with_timestamp(f"{self.name} shuffled for new epoch")
        self.batch_count = 0

    def _augment(self, frame):
        t = tf.convert_to_tensor(frame, dtype=tf.float32)
        t = tf.image.random_flip_left_right(t)
        t = tf.image.random_brightness(t, 0.1)
        t = tf.image.random_contrast(t, 0.9, 1.1)
        t = tf.image.rot90(t, k=tf.random.uniform([],maxval=4,dtype=tf.int32))
        return t.numpy()

# --- Model Builders ---
def build_efficientnet(img_size, mixed):
    log_with_timestamp("Building EfficientNetB0")
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet',
                                                input_shape=(img_size,img_size,3))
    base.trainable=False
    inp = tf.keras.Input((img_size,img_size,3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32' if mixed else None)(x)
    model = models.Model(inp,out)
    log_with_timestamp(f"EfficientNet parameters: {model.count_params():,}")
    return model

def build_resnet(img_size, mixed):
    log_with_timestamp("Building ResNet50")
    base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                        input_shape=(img_size,img_size,3))
    base.trainable=False
    inp = tf.keras.Input((img_size,img_size,3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32' if mixed else None)(x)
    model = models.Model(inp,out)
    log_with_timestamp(f"ResNet parameters: {model.count_params():,}")
    return model

def build_custom(img_size, mixed):
    log_with_timestamp("Building Custom CNN")
    inp = tf.keras.Input((img_size,img_size,3))
    x = layers.Conv2D(32,3,activation='relu',padding='same')(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512,activation='relu')(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256,activation='relu')(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32' if mixed else None)(x)
    model = models.Model(inp,out)
    log_with_timestamp(f"Custom CNN parameters: {model.count_params():,}")
    return model

# --- Optuna Objective ---
def objective(trial, train_gen, val_gen, cfg, mixed):
    log_with_timestamp(f"Optuna trial {trial.number} start")
    model_type = trial.suggest_categorical('model_type', ['efficientnet','resnet','custom'])
    lr = trial.suggest_float('learning_rate',1e-5,1e-2,log=True)
    log_with_timestamp(f"  params: model={model_type}, lr={lr:.5f}")
    if model_type=='efficientnet':
        model = build_efficientnet(cfg.img_size, mixed)
    elif model_type=='resnet':
        model = build_resnet(cfg.img_size, mixed)
    else:
        model = build_custom(cfg.img_size, mixed)
    opt = optimizers.Adam(learning_rate=lr)
    if mixed: opt = mixed_precision.LossScaleOptimizer(opt)
    model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy',
            tf.keras.metrics.AUC(name='auc')])
    es = callbacks.EarlyStopping('val_auc',patience=3,restore_best_weights=True,mode='max')
    hist = model.fit(train_gen, validation_data=val_gen,
                    epochs=10, callbacks=[es], verbose=1)
    auc = max(hist.history['val_auc'])
    tf.keras.backend.clear_session(); gc.collect()
    return 1-auc

# --- Progress Callback ---
class DetailedProgressCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        log_with_timestamp(f"Epoch {epoch+1}/{self.params['epochs']} begin")
    def on_epoch_end(self, epoch, logs=None):
        log_with_timestamp(f"Epoch {epoch+1} end: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, val_loss={logs['val_loss']:.4f}, val_acc={logs['val_accuracy']:.4f}")

# --- Main ---
def main():
    global logger
    cfg = get_config()
    logger = setup_logging(cfg.log_dir)
    start = time.time()
    log_with_timestamp("=== TRAINING START ===")
    gpu_ok = setup_rtx_3060_ti()
    mixed = setup_mixed_precision() and gpu_ok
    log_with_timestamp(f"Config: epochs={cfg.epochs}, batch={cfg.batch_size}, img={cfg.img_size}")
    files = sorted(glob.glob(os.path.join(cfg.data_dir,'*_frames.npy')))
    if not files: 
        log_with_timestamp(f"ERROR: no batches in {cfg.data_dir}"); return
    # load or split indices...
    # load or split indices...
    # Attempt to load existing train/val/test splits, else create and save new ones
    idx_train_path = os.path.join(cfg.data_dir, 'train_indices.npy')
    idx_val_path   = os.path.join(cfg.data_dir, 'val_indices.npy')
    idx_test_path  = os.path.join(cfg.data_dir, 'test_indices.npy')

    if all(os.path.exists(p) for p in (idx_train_path, idx_val_path, idx_test_path)):
        log_with_timestamp("Loading existing train/val/test splits")
        train_indices = np.load(idx_train_path)
        val_indices   = np.load(idx_val_path)
        test_indices  = np.load(idx_test_path)

        train_files = [files[i] for i in train_indices if i < len(files)]
        val_files   = [files[i] for i in val_indices   if i < len(files)]
        test_files  = [files[i] for i in test_indices  if i < len(files)]
    else:
        log_with_timestamp("Split indices not found; creating new 80/10/10 split")
        np.random.shuffle(files)
        N = len(files)
        n_train = int(0.8 * N)
        n_val   = int(0.9 * N)

        train_files = files[:n_train]
        val_files   = files[n_train:n_val]
        test_files  = files[n_val:]

        # Save indices for reproducibility
        np.save(idx_train_path, np.arange(0, n_train))
        np.save(idx_val_path,   np.arange(n_train, n_val))
        np.save(idx_test_path,  np.arange(n_val, N))
        log_with_timestamp(f"Saved train/val/test indices to {cfg.data_dir}")

    log_with_timestamp(f"Data split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    train_gen = DeepfakeDataGenerator(train_files, cfg.batch_size, cfg.img_size, True, True, "TrainGen")
    val_gen   = DeepfakeDataGenerator(val_files,   cfg.batch_size, cfg.img_size, False, False, "ValGen")
    test_gen  = DeepfakeDataGenerator(test_files,  cfg.batch_size, cfg.img_size, False, False, "TestGen")

    if not cfg.skip_optuna:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t,train_gen,val_gen,cfg,mixed), n_trials=cfg.optuna_trials)
        best = study.best_params
    else:
        best = {'model_type':'efficientnet','learning_rate':0.001,'dropout_rate':0.3}
    log_with_timestamp(f"Best params: {best}")
    # build final model
    if best['model_type']=='efficientnet': model = build_efficientnet(cfg.img_size,mixed)
    elif best['model_type']=='resnet':         model = build_resnet(cfg.img_size,mixed)
    else:                                       model = build_custom(cfg.img_size,mixed)
    opt = optimizers.Adam(best['learning_rate'])
    if mixed: opt = mixed_precision.LossScaleOptimizer(opt)
    model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()])

    cbs = [
        DetailedProgressCallback(),
        callbacks.EarlyStopping('val_auc',patience=5,restore_best_weights=True,mode='max'),
        callbacks.ModelCheckpoint(os.path.join(cfg.model_dir,'best_model.keras'),
                                save_best_only=True,monitor='val_auc',mode='max'),
        callbacks.ReduceLROnPlateau('val_loss',factor=0.5,patience=3,min_lr=1e-7),
        callbacks.CSVLogger(os.path.join(cfg.log_dir,'training_log.csv')),
        callbacks.TensorBoard(log_dir=cfg.log_dir)
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=cfg.epochs, callbacks=cbs, verbose=1)
    log_with_timestamp("Evaluating on test set...")
    res = model.evaluate(test_gen, verbose=1)
    results = dict(test_loss=float(res[0]), test_acc=float(res[1]), test_auc=float(res[2]),
                test_prec=float(res[3]), test_rec=float(res[4]),
                total_time=time.time()-start, best_params=best)
    with open(os.path.join(cfg.log_dir,'final_results.json'),'w') as f: json.dump(results,f,indent=2)
    model.save(os.path.join(cfg.model_dir,'final_model.keras'))
    log_with_timestamp(f"=== TRAINING COMPLETE in {(time.time()-start)/60:.1f}m ===")

if __name__=="__main__":
    main()
