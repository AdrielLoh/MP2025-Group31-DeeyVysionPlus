import subprocess
import os
import math
import random

INPUT_DIR = "F:/MP-Training-Datasets/ryerson-av-REAL"
OUTPUT_DIR = "D:/model_training/cache/batches/physio-deep-v1/real"
PROCESSED = "D:/model_training/cache/batches/physio-deep-v1/real/real_processed.txt"
FAILED = "D:/model_training/cache/batches/physio-deep-v1/real/real_failed.txt"
LABEL = "real"
BATCH_SIZE = 40   # HDF5 videos per batch
BATCHES_PER_RUN = 3   # Number of HDF5 files per subprocess
STOP_FLAG_PATH = "model_training_stuff/physio_model_deeplearning/stop.flag"   # Change this path as needed

def get_video_files(input_dir, exts=('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(exts)]

def load_txt_set(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

# Gather video files and exclude processed/failed
videos = get_video_files(INPUT_DIR)
processed = load_txt_set(PROCESSED)
failed = load_txt_set(FAILED)
exclude = processed.union(failed)

videos = [v for v in videos if os.path.basename(v) not in exclude]
random.shuffle(videos)

total_videos = len(videos)
videos_per_run = BATCH_SIZE * BATCHES_PER_RUN

for i in range(0, total_videos, videos_per_run):
    # --- Check for stop.flag BEFORE launching the next batch ---
    if os.path.exists(STOP_FLAG_PATH):
        print(f"Detected {STOP_FLAG_PATH}, stopping script.")
        break

    chunk = videos[i:i+videos_per_run]
    chunk_file = f"video_chunk_{i//videos_per_run}.txt"
    with open(chunk_file, 'w', encoding='utf-8') as f:
        for v in chunk:
            f.write(f"{v}\n")
    print(f"Launching batch {i//videos_per_run+1} with {len(chunk)} videos...")
    subprocess.run([
        "python", "model_training_stuff/physio_model_deeplearning/preprocessing-rewritten.py",
        "--input", chunk_file,
        "--output", OUTPUT_DIR,
        "--label", LABEL,
        "--batch_size", str(BATCH_SIZE),
        "--max_workers", "8"
    ])
    os.remove(chunk_file)
