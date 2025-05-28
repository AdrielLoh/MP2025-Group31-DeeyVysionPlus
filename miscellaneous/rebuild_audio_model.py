import os
import tensorflow as tf
import joblib
from tensorflow.keras import regularizers

# Constants
INPUT_HEIGHT = 170
INPUT_WIDTH = 157
INPUT_CHANNELS = 1
MODEL_WEIGHTS_PATH = "models/audio_model_alcj_weights_v2p1.pkl"  # Saved weights from my model built in TF 2.10
RESAVED_MODEL_PATH = "models/audio_model_tf217_alcj_v2.keras" # Rebuilt model compatible with current project

# TCN residual block (basically a CNN-style LSTM)
def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    prev_x = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
    if prev_x.shape[-1] != x.shape[-1]:
        prev_x = tf.keras.layers.Conv1D(filters, 1, padding='same')(prev_x)
    x = tf.keras.layers.Add()([x, prev_x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# CNN Stack
def build_model():
    inputs = tf.keras.layers.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)
    for dilation_rate in [1, 2, 4, 8]:
        x = residual_block(x, filters=128, kernel_size=3, dilation_rate=dilation_rate, dropout_rate=0.2)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs, outputs)

    return model

# Rebuild & load weights
model = build_model()
weights = joblib.load(MODEL_WEIGHTS_PATH)
model.set_weights(weights)

print("Weights loaded successfully.")

# Save in new keras for compatibility with TF 2.17+
model.save(RESAVED_MODEL_PATH)
print(f"Model re-saved to: {RESAVED_MODEL_PATH}")
