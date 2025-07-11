#
#
# THIS SCRIPT IS INTENDED FOR VERIFYING WHETHER A MODEL LOADS IN THE PROJECT
#
#
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras

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

model_path = 'models/audio_model_v9p2ft2.keras'

model = load_model(model_path, compile=False) # Replace the path with whatever model you want to verify
model.summary()