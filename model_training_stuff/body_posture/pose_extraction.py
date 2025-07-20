import os
import cv2
import numpy as np
import openpifpaf

# === Preprocessing Enhancements ===
def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    gamma = 1.2
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(enhanced, lut)

    wb = cv2.xphoto.createSimpleWB()
    balanced = wb.balanceWhite(corrected)

    return balanced

# === Pose Sequence Extraction ===
def extract_pose_sequence(video_path, max_frames=60):
    predictor = openpifpaf.Predictor(checkpoint='resnet50')
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_count = 0

    print(f"🔍 Processing: {os.path.basename(video_path)}")

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions, _, _ = predictor.numpy_image(rgb)

        if predictions:
            ann = predictions[0]
            keypoints = [coord for x, y, v in ann.data for coord in (x, y)]
            sequence.append(keypoints)
        else:
            sequence.append([0] * 34)  # Fallback: 17 keypoints × (x, y)

        frame_count += 1
        print(f"  🧠 Frame {frame_count}: {'Pose found' if predictions else 'No pose'}")

    cap.release()
    return np.array(sequence)

# === Process Directory Recursively and Save ===
def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                full_path = os.path.join(root, file)
                
                # Extract pose sequence
                sequence = extract_pose_sequence(full_path)

                # Construct relative path to preserve subdirectory structure
                rel_path = os.path.relpath(full_path, input_dir)
                save_path = os.path.join(output_dir, rel_path)
                save_dir = os.path.dirname(save_path)

                os.makedirs(save_dir, exist_ok=True)
                save_path_npy = os.path.splitext(save_path)[0] + '.npy'

                # Save the .npy file
                np.save(save_path_npy, sequence)
                print(f"✅ Saved to: {save_path_npy}\n")

# === Usage ===
if __name__ == "__main__":
    input_dir = 'C:/Training Videos/test'      # Folder of .mp4 videos
    output_dir = 'C:/Training Videos/real_poses'      # Folder for saved .npy pose files
    process_directory(input_dir, output_dir)
