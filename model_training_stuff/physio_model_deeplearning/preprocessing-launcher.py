import subprocess
import os
import math

INPUT_DIR = "G:/deepfake_training_datasets/Physio_Model/VALIDATION/real"
OUTPUT_DIR = "D:/model_training/cache/batches/physio-deep-v1/real"
LABEL = "real"
BATCH_SIZE = 40   # HDF5 videos per batch
BATCHES_PER_RUN = 5   # Number of HDF5 files per subprocess

def get_video_files(input_dir, exts=('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(exts)]

videos = get_video_files(INPUT_DIR)
total_videos = len(videos)
videos_per_run = BATCH_SIZE * BATCHES_PER_RUN

for i in range(0, total_videos, videos_per_run):
    chunk = videos[i:i+videos_per_run]
    # Save chunk to a temp file
    chunk_file = f"video_chunk_{i//videos_per_run}.txt"
    with open(chunk_file, 'w') as f:
        for v in chunk:
            f.write(f"{v}\n")
    print(f"Launching batch {i//videos_per_run+1} with {len(chunk)} videos...")
    # Launch subprocess for this chunk
    subprocess.run([
        "python", "model_training_stuff\physio_model_deeplearning\preprocessing-rewritten.py",
        "--input", chunk_file,
        "--output", OUTPUT_DIR,
        "--label", LABEL,
        "--batch_size", str(BATCH_SIZE),
        "--max_workers", "6"   # or whatever is safe for your RAM
    ])
    # Optionally, remove the temp file after
    os.remove(chunk_file)