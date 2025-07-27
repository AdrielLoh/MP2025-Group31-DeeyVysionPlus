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

# Configure enhanced logging with timestamps
def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_progress.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def log_with_timestamp(message):
    """Enhanced logging with immediate output"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)
    logger.info(message)

# --- GPU Setup for RTX 3060 Ti ---
def setup_rtx_3060_ti():
    """Setup RTX 3060 Ti with optimal memory management"""
    log_with_timestamp("Setting up RTX 3060 Ti GPU...")
    try:
        # UPDATED API - Removed experimental
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log_with_timestamp(f"Found {len(gpus)} GPU(s)")
            
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=7168)]
                )
                log_with_timestamp("GPU memory limited to 7GB")
            except Exception as mem_e:
                log_with_timestamp(f"Could not set memory limit: {mem_e}")
        else:
            log_with_timestamp("WARNING: No GPU found, using CPU")
        return True
    except Exception as e:
        log_with_timestamp(f"GPU setup failed: {e}")
        return False


def setup_mixed_precision():
    """Setup mixed precision for RTX 3060 Ti"""
    log_with_timestamp("Setting up mixed precision training...")
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        log_with_timestamp("Mixed precision (float16) enabled successfully")
        return True
    except Exception as e:
        log_with_timestamp(f"Mixed precision setup failed: {e}")
        return False

def force_cpu_only():
    """Force CPU-only training for Windows stability"""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    log_with_timestamp("Forced CPU-only mode for Windows stability")

# --- Configuration ---
def get_config():
    """Configuration optimized for RTX 3060 Ti (8GB VRAM)"""
    log_with_timestamp("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--data_dir', type=str, default='C:/deepvysion/preprocessed-1',
                        help='Directory containing preprocessed batch files')
    parser.add_argument('--model_dir', type=str, default='C:/deepvysion/deepfake_models',
                        help='Directory to save trained models')
    parser.add_argument('--log_dir', type=str, default='C:/deepvysion/logs',
                        help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,  # Optimized for 8GB VRAM
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Input image size')
    parser.add_argument('--optuna_trials', type=int, default=10,
                        help='Number of Optuna trials')
    parser.add_argument('--skip_optuna', action='store_true',
                        help='Skip hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Create directories
    log_with_timestamp(f"Creating output directories...")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_with_timestamp(f"Model directory: {args.model_dir}")
    log_with_timestamp(f"Log directory: {args.log_dir}")
    
    return args

# --- Enhanced Data Loading from Preprocessed Batches ---
class DeepfakeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_files, batch_size=16, img_size=128, shuffle=True, augment=False, name="DataGenerator"):
        self.batch_files = batch_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.name = name
        self.indexes = np.arange(len(self.batch_files))
        self.batch_count = 0
        log_with_timestamp(f"Initialized {self.name} with {len(self.batch_files)} batches, augment={self.augment}")
        self.on_epoch_end()

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, index):
        start_time = time.time()
        self.batch_count += 1
        
        log_with_timestamp(f"{self.name}: Processing batch {self.batch_count}/{len(self.batch_files)} (index={index})")
        
        batch_file = self.batch_files[self.indexes[index]]
        batch_name = os.path.basename(batch_file)
        
        log_with_timestamp(f"{self.name}: Loading {batch_name}...")
        
        try:
            # Load frames and labels from batch file
            frames = np.load(batch_file.replace('_frames.npy', '_frames.npy'), allow_pickle=True)
            labels = np.load(batch_file.replace('_frames.npy', '_labels.npy'), allow_pickle=True)
            
            log_with_timestamp(f"{self.name}: Loaded {len(frames)} videos from {batch_name}")
            
            # Process frames
            X = []
            y = []
            
            valid_videos = 0
            for video_idx, (video_frames, label) in enumerate(zip(frames, labels)):
                if video_frames is not None and len(video_frames) > 0:
                    valid_videos += 1
                    # Sample random frames from video
                    num_frames = min(len(video_frames), 8)  # Limit frames per video
                    selected_indices = np.random.choice(len(video_frames), num_frames, replace=False)
                    
                    for idx in selected_indices:
                        frame = video_frames[idx]
                        if frame is not None and frame.shape == (self.img_size, self.img_size, 3):
                            if self.augment:
                                frame = self.augment_frame(frame)
                            X.append(frame)
                            y.append(label)
            
            log_with_timestamp(f"{self.name}: Processed {valid_videos} valid videos from {batch_name}")
            
            # Convert to arrays and ensure proper batch size
            if len(X) == 0:
                log_with_timestamp(f"{self.name}: WARNING - Empty batch {batch_name}, returning zeros")
                return np.zeros((0, self.img_size, self.img_size, 3), dtype=np.float32), np.array([])
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            # Sample batch_size items if we have more
            if len(X) > self.batch_size:
                indices = np.random.choice(len(X), self.batch_size, replace=False)
                X = X[indices]
                y = y[indices]
            
            processing_time = time.time() - start_time
            log_with_timestamp(f"{self.name}: Completed batch {self.batch_count} in {processing_time:.2f}s, shape={X.shape}, real/fake split: {np.sum(y==0)}/{np.sum(y==1)}")
            
            return X, y
            
        except Exception as e:
            log_with_timestamp(f"{self.name}: ERROR processing {batch_name}: {e}")
            return np.zeros((0, self.img_size, self.img_size, 3), dtpe=np.float32), np.array([])

    def augment_frame(self, frame):
        """Apply data augmentation"""
        # Convert to tensor for TF operations
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        
        # Random flip
        frame = tf.image.random_flip_left_right(frame)
        
        # Random brightness and contrast
        frame = tf.image.random_brightness(frame, max_delta=0.1)
        frame = tf.image.random_contrast(frame, 0.9, 1.1)
        
        # Random rotation (small angle)
        frame = tf.image.rot90(frame, k=tf.random.uniform([], maxval=4, dtype=tf.int32))
        
        return frame.numpy()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            log_with_timestamp(f"{self.name}: Shuffled batch indices for new epoch")
        self.batch_count = 0  # Reset batch counter for new epoch

# --- Enhanced Model Architectures ---
def build_efficientnet_model(img_size=128, use_mixed_precision=False):
    """Build EfficientNet-based deepfake detection model"""
    log_with_timestamp("Building EfficientNetB0 model...")
    
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    log_with_timestamp(f"EfficientNetB0 base model loaded, trainable={base_model.trainable}")
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    if use_mixed_precision:
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    log_with_timestamp(f"EfficientNet model built successfully with {model.count_params():,} parameters")
    return model

def build_resnet_model(img_size=128, use_mixed_precision=False):
    """Build ResNet-based deepfake detection model"""
    log_with_timestamp("Building ResNet50 model...")
    
    base_model = tf.keras.applications.ResNet50(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    log_with_timestamp(f"ResNet50 base model loaded, trainable={base_model.trainable}")
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    if use_mixed_precision:
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    log_with_timestamp(f"ResNet model built successfully with {model.count_params():,} parameters")
    return model

def build_custom_cnn(img_size=128, use_mixed_precision=False):
    """Build custom CNN optimized for RTX 3060 Ti"""
    log_with_timestamp("Building custom CNN model...")
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    
    # Feature extraction layers
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    if use_mixed_precision:
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    log_with_timestamp(f"Custom CNN built successfully with {model.count_params():,} parameters")
    return model

# --- Enhanced Optuna Optimization ---
def objective(trial, train_gen, val_gen, config, use_mixed_precision):
    """Optuna objective function with progress logging"""
    trial_start_time = time.time()
    log_with_timestamp(f"Starting Optuna trial {trial.number}")
    
    try:
        # Hyperparameters to optimize
        model_type = trial.suggest_categorical('model_type', ['efficientnet', 'resnet', 'custom_cnn'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        
        log_with_timestamp(f"Trial {trial.number}: model={model_type}, lr={learning_rate:.6f}, dropout={dropout_rate:.3f}")
        
        # Build model based on type
        if model_type == 'efficientnet':
            model = build_efficientnet_model(config.img_size, use_mixed_precision)
        elif model_type == 'resnet':
            model = build_resnet_model(config.img_size, use_mixed_precision)
        else:
            model = build_custom_cnn(config.img_size, use_mixed_precision)
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        log_with_timestamp(f"Trial {trial.number}: Starting training (10 epochs)...")
        
        # Train with early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc', patience=3, restore_best_weights=True, mode='max'
        )
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,  # Shorter for hyperparameter search
            callbacks=[early_stopping],
            verbose=1  # Show progress bars for each epoch
        )
        
        # Get best validation AUC
        best_auc = max(history.history['val_auc'])
        trial_time = time.time() - trial_start_time
        
        log_with_timestamp(f"Trial {trial.number} completed in {trial_time:.2f}s, best AUC: {best_auc:.4f}")
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
        return 1.0 - best_auc  # Optuna minimizes
        
    except Exception as e:
        log_with_timestamp(f"Trial {trial.number} failed: {e}")
        tf.keras.backend.clear_session()
        gc.collect()
        return 1.0

# --- Enhanced Progress Callback ---
class DetailedProgressCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        
    def on_train_begin(self, logs=None):
        log_with_timestamp(f"Training started - {self.params['epochs']} epochs, {self.params['steps']} steps per epoch")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        log_with_timestamp(f"Starting epoch {epoch + 1}/{self.params['epochs']}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        log_with_timestamp(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        log_with_timestamp(f"Metrics - Loss: {logs.get('loss', 0):.4f}, Accuracy: {logs.get('accuracy', 0):.4f}, Val Loss: {logs.get('val_loss', 0):.4f}, Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
        
    def on_train_batch_begin(self, batch, logs=None):
        if batch % 10 == 0:  # Log every 10 batches
            log_with_timestamp(f"Processing training batch {batch + 1}/{self.params['steps']}")
            
    def on_test_batch_begin(self, batch, logs=None):
        if batch % 10 == 0:  # Log every 10 batches
            log_with_timestamp(f"Processing validation batch {batch + 1}")

# --- Main Training Function ---
def main():
    main_start_time = time.time()
    log_with_timestamp("=== DEEPFAKE DETECTION TRAINING STARTED ===")
    
    # Setup
    config = get_config()
    gpu_success = setup_rtx_3060_ti()
    use_mixed_precision = setup_mixed_precision() and gpu_success
    
    log_with_timestamp(f"Configuration: epochs={config.epochs}, batch_size={config.batch_size}, img_size={config.img_size}")
    log_with_timestamp(f"GPU setup: {gpu_success}, Mixed precision: {use_mixed_precision}")
    
    # Find batch files
    log_with_timestamp(f"Searching for batch files in {config.data_dir}...")
    batch_files = sorted(glob.glob(os.path.join(config.data_dir, '*_frames.npy')))
    if not batch_files:
        log_with_timestamp(f"ERROR: No batch files found in {config.data_dir}")
        return
    
    log_with_timestamp(f"Found {len(batch_files)} batch files")
    
    # Load split indices if they exist
    log_with_timestamp("Loading data split indices...")
    train_indices_file = os.path.join(config.data_dir, 'train_indices.npy')
    val_indices_file = os.path.join(config.data_dir, 'val_indices.npy')
    test_indices_file = os.path.join(config.data_dir, 'test_indices.npy')
    
    if all(os.path.exists(f) for f in [train_indices_file, val_indices_file, test_indices_file]):
        train_indices = np.load(train_indices_file)
        val_indices = np.load(val_indices_file)
        test_indices = np.load(test_indices_file)
        
        train_files = [batch_files[i] for i in train_indices if i < len(batch_files)]
        val_files = [batch_files[i] for i in val_indices if i < len(batch_files)]
        test_files = [batch_files[i] for i in test_indices if i < len(batch_files)]
        
        log_with_timestamp("Loaded existing data splits")
    else:
        # Simple split if indices don't exist
        log_with_timestamp("WARNING: Split indices not found, creating simple 80/10/10 split")
        np.random.shuffle(batch_files)
        n_train = int(0.8 * len(batch_files))
        n_val = int(0.9 * len(batch_files))
        
        train_files = batch_files[:n_train]
        val_files = batch_files[n_train:n_val]
        test_files = batch_files[n_val:]
    
    log_with_timestamp(f"Data split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create data generators
    log_with_timestamp("Creating data generators...")
    train_gen = DeepfakeDataGenerator(
        train_files, batch_size=config.batch_size, 
        img_size=config.img_size, shuffle=True, augment=True, name="TrainGen"
    )
    val_gen = DeepfakeDataGenerator(
        val_files, batch_size=config.batch_size, 
        img_size=config.img_size, shuffle=False, augment=False, name="ValGen"
    )
    test_gen = DeepfakeDataGenerator(
        test_files, batch_size=config.batch_size, 
        img_size=config.img_size, shuffle=False, augment=False, name="TestGen"
    )
    
    # Hyperparameter optimization
    best_params = None
    if not config.skip_optuna:
        log_with_timestamp(f"Starting hyperparameter optimization with {config.optuna_trials} trials...")
        
        def optuna_objective(trial):
            return objective(trial, train_gen, val_gen, config, use_mixed_precision)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(optuna_objective, n_trials=config.optuna_trials)
        
        best_params = study.best_params
        log_with_timestamp(f"Optuna optimization completed. Best hyperparameters: {best_params}")
        
        with open(os.path.join(config.log_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)
    else:
        # Default parameters optimized for RTX 3060 Ti
        best_params = {
            'model_type': 'efficientnet',
            'learning_rate': 0.001,
            'dropout_rate': 0.3
        }
        log_with_timestamp(f"Skipping Optuna, using default parameters: {best_params}")
    
    # Build final model
    log_with_timestamp("Building final model for training...")
    if best_params['model_type'] == 'efficientnet':
        model = build_efficientnet_model(config.img_size, use_mixed_precision)
    elif best_params['model_type'] == 'resnet':
        model = build_resnet_model(config.img_size, use_mixed_precision)
    else:
        model = build_custom_cnn(config.img_size, use_mixed_precision)
    
    # Compile model
    log_with_timestamp("Compiling model...")
    optimizer = optimizers.Adam(learning_rate=best_params['learning_rate'])
    if use_mixed_precision:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    log_with_timestamp("Model compiled successfully")
    
    # Callbacks
    log_with_timestamp("Setting up training callbacks...")
    callbacks_list = [
        DetailedProgressCallback(),
        callbacks.EarlyStopping(
            monitor='val_auc', patience=5, restore_best_weights=True, mode='max'
        ),
        callbacks.ModelCheckpoint(
            os.path.join(config.model_dir, 'best_model.keras'),
            save_best_only=True, monitor='val_auc', mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        callbacks.CSVLogger(os.path.join(config.log_dir, 'training_log.csv')),
        callbacks.TensorBoard(log_dir=config.log_dir, histogram_freq=1, update_freq='epoch')
    ]
    
    # Train model
    log_with_timestamp(f"Starting final training for {config.epochs} epochs...")
    training_start_time = time.time()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    training_time = time.time() - training_start_time
    log_with_timestamp(f"Initial training completed in {training_time:.2f}s")
    
    # Fine-tuning (unfreeze base model)
    if best_params['model_type'] in ['efficientnet', 'resnet']:
        log_with_timestamp("Starting fine-tuning with unfrozen base model...")
        
        # Unfreeze base model
        model.layers[1].trainable = True  # Assuming base model is at index 1
        log_with_timestamp(f"Base model unfrozen, now trainable={model.layers[1].trainable}")
        
        # Lower learning rate for fine-tuning
        fine_tune_lr = best_params['learning_rate'] * 0.1
        log_with_timestamp(f"Fine-tuning with reduced learning rate: {fine_tune_lr:.6f}")
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        # Fine-tune for a few epochs
        fine_tune_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=5,
            callbacks=[
                DetailedProgressCallback(),
                callbacks.EarlyStopping(monitor='val_auc', patience=2, mode='max')
            ],
            verbose=1
        )
        
        log_with_timestamp("Fine-tuning completed")
    
    # Final evaluation
    log_with_timestamp("Starting final evaluation on test set...")
    eval_start_time = time.time()
    test_results = model.evaluate(test_gen, verbose=1)
    eval_time = time.time() - eval_start_time
    
    log_with_timestamp(f"Test evaluation completed in {eval_time:.2f}s")
    
    # Save results
    results = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_auc': float(test_results[2]),
        'test_precision': float(test_results[3]),
        'test_recall': float(test_results[4]),
        'best_params': best_params,
        'config': vars(config),
        'total_training_time': time.time() - main_start_time
    }
    
    results_file = os.path.join(config.log_dir, 'final_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_with_timestamp(f"Results saved to: {results_file}")
    
    # Save model
    model_file = os.path.join(config.model_dir, 'final_model.keras')
    model.save(model_file)
    log_with_timestamp(f"Final model saved to: {model_file}")
    
    total_time = time.time() - main_start_time
    log_with_timestamp("=== TRAINING COMPLETED SUCCESSFULLY ===")
    log_with_timestamp(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    log_with_timestamp(f"Test Accuracy: {results['test_accuracy']:.4f}")
    log_with_timestamp(f"Test AUC: {results['test_auc']:.4f}")
    log_with_timestamp(f"All results and logs saved to: {config.log_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_with_timestamp("Training interrupted by user")
    except Exception as e:
        log_with_timestamp(f"Training failed with error: {e}")
        import traceback
        log_with_timestamp(f"Full traceback: {traceback.format_exc()}")
        raise
