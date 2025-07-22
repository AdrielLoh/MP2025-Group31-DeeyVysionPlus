import os
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score
from datetime import datetime

# === 🗂️ Dataset: Load from Directories ===
class PoseSequenceDataset(Dataset):
    def __init__(self, root_dir, max_frames=60, zero_threshold=0.7, balance=True):
        self.data = []
        self.labels = []
        self.label_map = {'real': 0, 'fake': 1}
        self.samples_by_label = {'real': [], 'fake': []}
        self.label_counts = {'real': 0, 'fake': 0}
        self.skipped_files = []

        for label_name in self.label_map:
            label_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_path):
                continue

            for root, _, files in os.walk(label_path):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        sequence = np.load(file_path)

                        # Reject if mostly zeros
                        zero_ratio = 1 - np.count_nonzero(sequence) / sequence.size
                        if zero_ratio >= zero_threshold:
                            self.skipped_files.append(file_path)
                            continue

                        # Pad or trim to fixed length
                        if sequence.shape[0] < max_frames:
                            pad = np.zeros((max_frames - sequence.shape[0], sequence.shape[1]))
                            sequence = np.vstack([sequence, pad])
                        else:
                            sequence = sequence[:max_frames]

                            self.samples_by_label[label_name].append((sequence, self.label_map[label_name]))

        # === Balance classes ===
        if balance:
            real_samples = self.samples_by_label['real']
            fake_samples = self.samples_by_label['fake']
            min_count = min(len(real_samples), len(fake_samples))

            # Undersample fakes
            fake_samples = random.sample(fake_samples, min_count)

            # Oversample reals (with replacement)
            real_samples = random.choices(real_samples, k=min_count)

            balanced_samples = real_samples + fake_samples
            random.shuffle(balanced_samples)
        else:
            balanced_samples = self.samples_by_label['real'] + self.samples_by_label['fake']

         # Finalize data
        self.data = [x[0] for x in balanced_samples]
        self.labels = [x[1] for x in balanced_samples]

        print("📊 Balanced Dataset Summary:")
        print(f"  └ Real: {sum(1 for l in self.labels if l == 0)}")
        print(f"  └ Fake: {sum(1 for l in self.labels if l == 1)}")
        print(f"🚫 Skipped: {len(self.skipped_files)}")
        print(f"🧪 Total samples: {len(self.data)}\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y



# === 🧠 Model: LSTM Classifier ===
class PoseClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=256, num_layers=2, num_classes=2):
        super(PoseClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# === 🔁 Training + Validation ===
def train_pose_model(data_dir, epochs=30, batch_size=32, max_frames=60, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    # Prepare data + validation split (80/20)
    full_dataset = PoseSequenceDataset(root_dir=data_dir, max_frames=max_frames)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Build model
    model = PoseClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))  # model weights
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"model_training_stuff/body_posture/checkpoints/pose_model_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # === 🔎 Validation ===
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_preds = val_outputs.argmax(dim=1)

                all_preds.extend(val_preds.cpu().numpy())
                all_targets.extend(val_targets.cpu().numpy())


                val_correct += (val_preds == val_targets).sum().item()
                val_total += val_targets.size(0)

        val_acc = val_correct / val_total
        # Precision for "real" class (label 0)
        precision_real = precision_score(all_targets, all_preds, pos_label=0)
        # Recall for "fake" class (label 1)
        recall_fake = recall_score(all_targets, all_preds, pos_label=1)
        print(f"📊 Epoch {epoch}/{epochs}:")
        print(f"  └ Train Loss: {avg_loss:.4f}")
        print(f"  └ Train Acc: {train_acc:.2%}")
        print(f"  └ Val Acc: {val_acc:.2%}")
        print(f"  └ Precision (real): {precision_real:.2%}")
        print(f"  └ Recall (fake): {recall_fake:.2%}")
        
        

        # === 💾 Save best model ===
        if val_acc > best_val_acc:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"✅ Best model saved at epoch {epoch} to {save_dir}/best_model.pth")
            best_val_acc = val_acc

    print("🏁 Training complete.")

# === 🚀 Start Training ===
if __name__ == "__main__":
    data_dir = "E:/deepfake videos/faceforensics/c23_poses"
    train_pose_model(data_dir)


