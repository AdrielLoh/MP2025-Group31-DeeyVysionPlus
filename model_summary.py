#
#
# THIS SCRIPT IS INTENDED FOR VERIFYING WHETHER A MODEL LOADS IN THE PROJECT
#
#

from tensorflow.keras.models import load_model

model = load_model("models/body_posture.keras") # Replace the path with whatever model you want to verify
model.summary()