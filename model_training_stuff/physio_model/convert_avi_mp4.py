import os
import subprocess

# === CONFIG ===
input_dir = "G:\deepfake_training_datasets\Physio_Model\DeepfakeTIMIT\higher_quality"  # Change this to your target folder
ffmpeg_path = "ffmpeg"  # If ffmpeg isn't in PATH, use full path like "C:/ffmpeg/bin/ffmpeg.exe"

# === Walk through all files recursively
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".avi"):
            avi_path = os.path.join(root, file)
            mp4_filename = os.path.splitext(file)[0] + ".mp4"
            mp4_path = os.path.join(root, mp4_filename)

            # Skip if the .mp4 already exists
            if os.path.exists(mp4_path):
                print(f"Skipping (already exists): {mp4_path}")
                continue

            # Build ffmpeg command
            command = [
                ffmpeg_path,
                "-i", avi_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-y",  # Overwrite output if needed
                mp4_path
            ]

            print(f"Converting: {avi_path} -> {mp4_path}")
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode == 0:
                print(f"Successfully converted. Deleting original: {avi_path}")
                os.remove(avi_path)
            else:
                print(f"❌ Failed to convert {avi_path}:\n{result.stderr.decode()}")
