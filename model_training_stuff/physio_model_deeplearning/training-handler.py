import subprocess
import time
import sys

# Number of allowed consecutive segfaults before giving up
MAX_ATTEMPTS = 100

PYTHON_EXEC = sys.executable  # Use current Python
TRAIN_SCRIPT = "model_training_stuff/physio_model_deeplearning/training.py"

TRAIN_ARGS = [
    "--data_dir", "/root/model_training/cache/batches/physio-deep-v1/for-training",
    "--model_dir", "/root/model_training/physiological-model/deep-learning-1-1",
    "--log_dir", "/root/model_training/physiological-model/deep-learning-1-1/logs",
    "--epochs", "100",
    "--batch_size", "32"
]

attempt = 0
while attempt < MAX_ATTEMPTS:
    print(f"\n[Watchdog] Starting training attempt {attempt + 1}...")
    process = subprocess.Popen([PYTHON_EXEC, TRAIN_SCRIPT] + TRAIN_ARGS)
    process.wait()
    exit_code = process.returncode

    if exit_code == 0:
        print("[Watchdog] Training finished successfully!")
        break
    elif exit_code == -11:
        print("[Watchdog] Segmentation fault detected. Restarting training...")
    else:
        print(f"[Watchdog] Training script exited with code {exit_code}. Exiting Now...")
        break

    attempt += 1
    time.sleep(10)  # Wait a bit before restarting to avoid thrashing

if attempt == MAX_ATTEMPTS:
    print("[Watchdog] Too many failures. Exiting.")

print("[Watchdog] Done.")
