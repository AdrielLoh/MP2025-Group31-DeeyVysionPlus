import subprocess
import signal 
import sys

# ==== This script is used to launch and manage the preprocessing for the Physio Model dataset. (like a watchdog) =====
"""
This script will be useful to automatically launch the main preprocessing script mutiple times to process different datasets.
It will also handle the subprocesses and allow for graceful termination on keyboard interrupts.
"""
def run_preprocessing(input_folder, output_folder, label, augment_chance, batch_size=40, max_workers=4):
    cmd = [
        "python", "model_training_stuff/physio_model_XGBoost/preprocessing_scripts_v2/main_preprocessing.py",
        "--input", input_folder,
        "--output", output_folder,
        "--label", label,
        "--augment_chance", str(augment_chance),
        "--batch_size", str(batch_size),
        "--max_workers", str(max_workers)
    ]
    print("Running:", " ".join(cmd))
    print("Running:", " ".join(cmd))
    try:
        # Use subprocess.Popen to allow signal handling for graceful termination
        proc = subprocess.Popen(cmd)
        def handle_sigint(sig, frame):
            print("\nReceived stop signal. Waiting for the current batch to finish and save...")
            proc.terminate()
        signal.signal(signal.SIGINT, handle_sigint)
        proc.wait()
        if proc.returncode == 0:
            print(f"Preprocessing for {label} finished normally.")
        else:
            print(f"Preprocessing for {label} exited with code {proc.returncode}.")
            if proc.returncode != -15:  # Note: "-15" is SIGTERM
                raise RuntimeError(f"Preprocessing for {label} failed!")
    except KeyboardInterrupt:
        print("Main script interrupted. Waiting for subprocess cleanup...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    # Edit these paths as needed
    fake_input = "E:/deepfake_training_datasets/Physio_Model/TRAINING/fake"
    real_input = "E:/deepfake_training_datasets/Physio_Model/TRAINING/real"
    # real_input_2 = "F:/MP-Training-Datasets/real-celebvhq/35666"
    real_input_2 = "E:/deepfake_training_datasets/Physio_Model/VALIDATION/real-semi-frontal"
    real_input_3 = "E:/deepfake_training_datasets/Physio_Model/TRAINING/real-semi-frontal"
    output_dir = "C:/model_training/physio_ml_22-7"

    # Run fake first (0.35 aug chance)
    run_preprocessing(
        input_folder=fake_input,
        output_folder=output_dir,
        label="fake",
        augment_chance=0.35,
        batch_size=100,
        max_workers=4
    )

    # Run real next (0.65 aug chance)
    run_preprocessing(
        input_folder=real_input,
        output_folder=output_dir,
        label="real",
        augment_chance=0.55,
        batch_size=100,
        max_workers=4
    )

    run_preprocessing(
        input_folder=real_input_3,
        output_folder=output_dir,
        label="real",
        augment_chance=0.5,
        batch_size=100,
        max_workers=4
    )
    run_preprocessing(
        input_folder=real_input_2,
        output_folder=output_dir,
        label="real",
        augment_chance=0.5,
        batch_size=100,
        max_workers=4
    )

    print("Both fake and real preprocessing complete.")
