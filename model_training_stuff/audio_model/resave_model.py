import tensorflow as tf

# Custom loss class (must match original)
class CustomFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        loss = alpha_t * tf.pow(1 - p_t, self.gamma) * bce

        return tf.reduce_mean(loss)

# Load model saved in 2.10
model = tf.keras.models.load_model(
    "models/training-8-fixed-compatability/stage-1/audio_model_alcj.keras",
    custom_objects={'CustomFocalLoss': CustomFocalLoss}
)
model.save("audio_model_alcj_resaved.h5")  # This works only in TF >= 2.11

print("Model successfully resaved")
