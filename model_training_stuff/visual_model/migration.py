import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model
import numpy as np
import glob

# automatically find input dimensions from a preprocessed batch 
sample_file = glob.glob('/mnt/d/preprocessed_data/*.npz')[0]
feat_dim = np.load(sample_file)['features'].shape[1]

def build_model(input_dim):
    inputs = layers.Input(shape=(input_dim,), dtype=tf.float32)
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
    return model

# build the model using detected input dimensions
model = build_model(feat_dim)

# load best weights (not final weights)
model.load_weights('model_outputs/best_weights.weights.h5')

# export as keras model for deployment in the website
model.save('visual_artifacts_model_migrated.keras')
print(f"Model exported with input_dim={feat_dim} as visual_artifacts_model_migrated.keras")
