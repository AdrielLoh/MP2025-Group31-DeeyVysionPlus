import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_training import PoseClassifier

# === 🧼 Clean Dataset Loader ===
class FilteredPoseDataset(Dataset):
    def __init__(self, root_dir, max_frames=60, zero_threshold=0.5):
        self.data = []
        self.labels = []
        self.label_map = {'real': 0, 'fake': 1}
        self.skipped = []

        for label in self.label_map:
            folder_path = os.path.join(root_dir, label)
            if not os.path.isdir(folder_path):
                continue

            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.npy'):
                        path = os.path.join(root, file)
                        try:
                            sequence = np.load(path)

                            # === Filter: skip if more than `zero_threshold` of elements are zero
                            zero_ratio = 1 - np.count_nonzero(sequence) / sequence.size
                            if zero_ratio >= zero_threshold:
                                self.skipped.append(path)
                                continue

                            # Pad or trim
                            if sequence.shape[0] < max_frames:
                                pad = np.zeros((max_frames - sequence.shape[0], sequence.shape[1]))
                                sequence = np.vstack([sequence, pad])
                            else:
                                sequence = sequence[:max_frames]

                            self.data.append(sequence)
                            self.labels.append(self.label_map[label])

                        except Exception as e:
                            print(f"⚠️ Error loading {path}: {e}")
                            self.skipped.append(path)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        # Summary
        print("📂 Evaluation Dataset Summary:")
        for label in self.label_map:
            count = sum(1 for i in self.labels if i == self.label_map[label])
            print(f"  └── {label}: {count} usable samples")
        print(f"🚫 Skipped due to low pose content: {len(self.skipped)}")
        print(f"🧪 Total samples for evaluation: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# === 🧠 Evaluation Function ===
def evaluate_model(model_path, data_dir, batch_size=32, max_frames=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating on device: {device}")

    dataset = FilteredPoseDataset(root_dir=data_dir, max_frames=max_frames)
    loader = DataLoader(dataset, batch_size=batch_size)

    if len(dataset) == 0:
        print("🚫 No usable data available.")
        return

    model = PoseClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    acc = correct / total if total > 0 else 0
    print(f"✅ Accuracy on filtered dataset: {acc:.2%} ({correct}/{total} correct)")

# === Run the Evaluation ===
if __name__ == "__main__":
    model_path = "model_training_stuff/body_posture/checkpoints/pose_model_20250714_233139/best_model.pth"  # Update with your actual path
    data_dir = "E:/deepfake videos/faceforensics/c23_poses"
    evaluate_model(model_path, data_dir)
