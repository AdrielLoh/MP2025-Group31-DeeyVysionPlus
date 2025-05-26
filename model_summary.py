#
#
# THIS SCRIPT IS INTENDED FOR VERIFYING WHETHER A MODEL LOADS IN THE PROJECT
#
#

from tensorflow.keras.models import load_model

model = load_model("models/audio_model_tf217_alcj.keras") # Replace the path with whatever model you want to verify
model.summary()