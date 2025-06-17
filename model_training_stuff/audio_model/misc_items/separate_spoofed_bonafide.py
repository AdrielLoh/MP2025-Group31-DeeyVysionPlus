import os
import shutil
import multiprocessing
from functools import partial

# CONFIGURATIONS
METADATA_PATH = "G:/deepfake_training_datasets/ASVspoof-2021/keys/DF/CM/trial_metadata.txt"
SOURCE_AUDIO_DIR = "G:/deepfake_training_datasets/ASVspoof-2021/ASVspoof2021_DF_eval/flac"  # folder where your DF_E_... files are
OUTPUT_DIR = "G:/deepfake_training_datasets/ASVspoof-2021/ASVspoof2021_DF_eval/separated"
SPOOF_DIR = os.path.join(OUTPUT_DIR, "spoofed")
BONAFIDE_DIR = os.path.join(OUTPUT_DIR, "bonafide")
NUM_PROCESSES = 6

# Ensure output folders exist
os.makedirs(SPOOF_DIR, exist_ok=True)
os.makedirs(BONAFIDE_DIR, exist_ok=True)

def process_line(line, source_dir, spoof_dir, bonafide_dir):
    parts = line.strip().split()
    if len(parts) < 3:
        return  # skip malformed lines

    filename = parts[1]  # e.g., DF_E_2000027
    label = parts[5].lower()  # should be either 'spoof' or 'bonafide'
    src_file = os.path.join(source_dir, f"{filename}.flac")  # Adjust extension if needed

    if not os.path.exists(src_file):
        return  # skip if file doesn't exist

    dst_dir = spoof_dir if label == "spoof" else bonafide_dir
    dst_path = os.path.join(dst_dir, f"{filename}.flac")

    try:
        shutil.move(src_file, dst_path)
        print(f"Moved: {filename} to {'spoofed' if label == 'spoof' else 'bonafide'}")
    except Exception as e:
        print(f"Failed to move {filename}: {e}")

if __name__ == "__main__":
    with open(METADATA_PATH, "r") as f:
        lines = [line for line in f if line.strip() and "DF_E_" in line]

    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        pool.map(
            partial(
                process_line,
                source_dir=SOURCE_AUDIO_DIR,
                spoof_dir=SPOOF_DIR,
                bonafide_dir=BONAFIDE_DIR,
            ),
            lines
        )

    print("Separation complete.")
