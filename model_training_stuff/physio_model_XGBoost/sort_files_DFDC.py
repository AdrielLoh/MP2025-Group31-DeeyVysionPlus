import os
import shutil
import re

src_folder = 'E:/deepfake_training_datasets/Physio_Model/TRAINING/fake-do-not-augment'      # <- Change this to your folder path
dst_folder = 'E:/deepfake_training_datasets/Physio_Model/TRAINING/deeperforensics-fake-unedited' # <- Change this to your dest path

os.makedirs(dst_folder, exist_ok=True)

# Regex: deeperforensics_number_M###.mp4 or deeperforensics_number_W###.mp4
pattern = re.compile(r'^(deeperforensics_\d+_(?:M|W)\d{3})\.mp4$', re.IGNORECASE)

for filename in os.listdir(src_folder):
    match = pattern.match(filename)
    if match:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")

print("Done!")
