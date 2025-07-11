"""
TO BE RUN IN TF 2.17+
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import joblib
import keras

weights_path = "model_training_stuff/physio_model_deeplearning/train_2_fold_1_weights.pkl"

@keras.saving.register_keras_serializable()
def masked_gap(args):
    f, m = args
    seq_len = tf.shape(f)[1]
    m = m[:, :seq_len]
    m = tf.cast(tf.expand_dims(m, -1), f.dtype)
    num = tf.reduce_sum(f * m, axis=1)
    denom = tf.reduce_sum(m, axis=1)
    pooled = num / tf.clip_by_value(denom, 1e-3, tf.float32.max)
    return pooled

def build_tcn_transformer(cfg, use_mixed_precision=False):
    window_size = cfg.get('window_size', 150)
    n_roi_features = cfg['n_roi_features']
    tcn_channels = cfg.get('filters', 48)
    ff_dim = cfg.get('dense_dim', 96)
    num_heads = 4  # you can tune this too
    attn_dropout = cfg.get('dropout', 0.2)
    tcn_blocks = [2 ** i for i in range(cfg['blocks'])]

    # Input layers
    roi_in = layers.Input((window_size, n_roi_features), name='roi_in')
    mask = layers.Input((window_size,), name='mask_in')

    # TCN backbone
    x = roi_in
    for d in tcn_blocks:
        res = x
        x = layers.Conv1D(tcn_channels, 3, padding="causal", dilation_rate=d, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(tcn_channels, 3, padding="causal", dilation_rate=d, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        # Projection for residual if needed
        if int(res.shape[-1]) != tcn_channels:
            res = layers.Conv1D(tcn_channels, 1, padding="same", kernel_regularizer=regularizers.l2(1e-4))(res)
        x = layers.Add()([res, x])
        x = layers.Activation("relu")(x)

    # Downsample and project
    x = layers.MaxPooling1D(2)(x)  # shape: (window_size//2, tcn_channels)
    x = layers.Conv1D(ff_dim, 1, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)  # shape: (window_size//2, ff_dim)

    # After MaxPooling1D and Conv1D
    seq_len = x.shape[1]
    pos_indices = tf.range(seq_len)
    # Create embedding weights as a tensor
    pos_emb_layer = layers.Embedding(input_dim=seq_len, output_dim=ff_dim)
    pos_emb = pos_emb_layer(pos_indices)
    pos_emb = tf.expand_dims(pos_emb, 0)  # (1, seq_len, ff_dim)
    x = layers.Add()([x, pos_emb])

    # Single Transformer block
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=ff_dim // num_heads,
        dropout=attn_dropout
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ffn = layers.Dense(ff_dim, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    ffn = layers.Dense(ff_dim, kernel_regularizer=regularizers.l2(1e-4))(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    pooled = layers.Lambda(masked_gap, name='masked_pooling')([x, mask])
    # print("pooled.shape:", pooled.shape)

    # Classification head
    dense = layers.Dense(ff_dim, kernel_regularizer=regularizers.l2(1e-4))(pooled)
    dense = layers.ReLU()(dense)
    dense = layers.Dropout(attn_dropout)(dense)
    dense = layers.Dense(ff_dim // 2, kernel_regularizer=regularizers.l2(1e-4))(dense)
    dense = layers.ReLU()(dense)
    dense = layers.Dropout(attn_dropout)(dense)
    if use_mixed_precision:
        out = layers.Dense(1, activation='sigmoid', dtype='float32', name='output', kernel_regularizer=regularizers.l2(1e-4))(dense)
    else:
        out = layers.Dense(1, activation='sigmoid', name='output', kernel_regularizer=regularizers.l2(1e-4))(dense)
    model = models.Model([roi_in, mask], out, name='TCN_Transformer')
    return model

try:
    # Default configuration
    model_config = {
        'blocks': 6,
        'filters': 64,
        'dense_dim': 128,
        'dropout': 0.3,
        'n_roi_features': 15,
        'window_size': 150
    }
    
    # Build model architecture
    model = build_tcn_transformer(model_config)
    
    # Load weights
    weights = joblib.load(weights_path)
    model.set_weights(weights)
    print("Weights loaded successfully.")

    # Save model to disk
    model.save("models/physio_tcn_transformer_2.keras")
    print("New model saved")
    
except Exception as e:
    print(f"Error: {e}")
    print("Failed to save new model")