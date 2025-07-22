import torch
from model_training import PoseClassifier  # Or define your model again if not in a separate file
import numpy as np

model = PoseClassifier()
checkpoint = torch.load("model_training_stuff/body_posture/checkpoints/pose_model_20250717_140239/best_model.pth", map_location='cpu')  # Adjust path
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def preprocess_sequence(npy_path, max_frames=60):
    sequence = np.load(npy_path)
    if sequence.shape[0] < max_frames:
        pad = np.zeros((max_frames - sequence.shape[0], sequence.shape[1]))
        sequence = np.vstack([sequence, pad])
    else:
        sequence = sequence[:max_frames]
    return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    input_seq = preprocess_sequence("E:/deepfake videos/faceforensics/c23_poses/real/actors/videos/03__talking_against_wall.npy")
    output = model(input_seq)
    prediction = torch.argmax(output, dim=1).item()
    label_map = {0: 'real', 1: 'fake'}
    print(f"🧠 Predicted label: {label_map[prediction]}")
