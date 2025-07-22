import os
import cv2
import numpy as np
import openpifpaf
import torch
from scipy.spatial.distance import cdist

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

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

# === Pose Sequence Extraction with Tracking ===
def extract_tracked_poses(video_path, max_distance=50):
    predictor = openpifpaf.Predictor(checkpoint='resnet50')
    cap = cv2.VideoCapture(video_path)
    person_tracks = {}       # person_id → list of poses
    person_centers = {}      # person_id → last known center
    next_id = 0

    print(f"Tracking: {os.path.basename(video_path)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions, _, _ = predictor.numpy_image(rgb)

        current_centers = []
        current_poses = []

        for ann in predictions:
            keypoints = [coord for x, y, v in ann.data for coord in (x, y)]
            if len(keypoints) == 34:
                current_poses.append(keypoints)
                xs = keypoints[::2]
                ys = keypoints[1::2]
                center = [np.mean(xs), np.mean(ys)]
                current_centers.append(center)

        assigned_ids = [-1] * len(current_poses)

        # Match current poses to existing tracked people
        if person_centers and current_centers:
            known_ids = list(person_centers.keys())
            known_centers = np.array([person_centers[pid] for pid in known_ids])
            distances = cdist(known_centers, np.array(current_centers))

            for known_idx, row in enumerate(distances):
                min_idx = np.argmin(row)
                if row[min_idx] < max_distance and assigned_ids[min_idx] == -1:
                    assigned_ids[min_idx] = known_ids[known_idx]

        # Assign poses to IDs
        for i, pose in enumerate(current_poses):
            if assigned_ids[i] != -1:
                pid = assigned_ids[i]
            else:
                pid = next_id
                next_id += 1
                person_tracks[pid] = []

            person_tracks[pid].append(pose)
            person_centers[pid] = current_centers[i]

    cap.release()
    return person_tracks


# === Process Directory Recursively and Save ===
def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                full_path = os.path.join(root, file)

                # Construct relative path to preserve subdirectory structure
                rel_path = os.path.relpath(full_path, input_dir)
                save_path = os.path.join(output_dir, rel_path)
                save_dir = os.path.dirname(save_path)
                base_name = os.path.splitext(save_path)[0]

                # Check if file already exists
                save_path_npy = os.path.splitext(save_path)[0] + '_poses.npy'
                if os.path.exists(save_path_npy):
                    print(f"Skipped: {save_path_npy} already exists")
                    continue

                os.makedirs(save_dir, exist_ok=True)

                # Extract tracked poses
                person_tracks = extract_tracked_poses(full_path)

                np.save(save_path_npy, person_tracks)
                print(f"Saved: {save_path_npy}")


# === Usage ===
input_dir = 'E:/deepfake videos/faceforensics/c23/manipulated'
output_dir = 'E:/deepfake videos/faceforensics/c23_poses/fake'
process_directory(input_dir, output_dir)
