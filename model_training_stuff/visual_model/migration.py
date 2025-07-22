import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# Define model architecture (must match the training architecture)
feat_dim = 1790  # replace with actual feature length (e.g., 1792 as computed before)
inputs = layers.Input(shape=(feat_dim,), dtype=tf.float32)
norm = layers.LayerNormalization(axis=1)(inputs)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L2(1e-4))(norm)
x = layers.Dropout(0.3)(x)  # dropout is ignored during inference but we include layer for completeness
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(1e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_2_17 = Model(inputs=inputs, outputs=outputs)

# Load weights from the best model (from training outputs)
best_weights_path = "./model_outputs/best_weights.weights.h5"
model_2_17.load_weights(best_weights_path)

# Save the model in Keras v3 format (TensorFlow 2.17+)
export_path = "./model_outputs/visual_artifactsv2.keras"
model_2_17.save(export_path)
print(f"Model successfully migrated and saved to {export_path}")
