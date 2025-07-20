import os
import json
import shutil

# === CONFIG ===
metadata_path = "G:/deepfake_training_datasets/Physio_Model/Deepfake Detection Challenge/train_sample_videos/metadata.json" # Replace with your actual path
video_folder = "G:/deepfake_training_datasets/Physio_Model/Deepfake Detection Challenge/train_sample_videos"     # Folder containing all the .mp4 videos
real_folder = "G:/deepfake_training_datasets/Physio_Model/TESTING/real"      # Destination for real videos
fake_folder = "G:/deepfake_training_datasets/Physio_Model/TESTING/fake"      # Destination for fake videos

# === Create output folders if they don't exist
os.makedirs(real_folder, exist_ok=True)
os.makedirs(fake_folder, exist_ok=True)

# === Load metadata
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# === Move files based on label
for filename, info in metadata.items():
    label = info.get("label")
    src_path = os.path.join(video_folder, filename)

    if not os.path.exists(src_path):
        print(f"❌ File not found: {src_path}")
        continue

    if label == "REAL":
        dest_path = os.path.join(real_folder, filename)
    elif label == "FAKE":
        dest_path = os.path.join(fake_folder, filename)
    else:
        print(f"⚠️ Unknown label for {filename}: {label}")
        continue

    # Move the file (use shutil.copy if you prefer copying)
    shutil.move(src_path, dest_path)
    print(f"Moved: {filename} → {dest_path}")

print("✅ Done sorting videos.")
