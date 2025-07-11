import os
import cv2
import numpy as np
import pandas as pd
import openpifpaf
import tensorflow as tf
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Configuration
PROCESSED_FOLDER = "static/processed/"
FEATURES_FOLDER = "static/features/"
MODEL_PATH_STATIC = 'models/body_posture.keras'
MODEL_PATH_LIVE = 'models/body_posture_live.keras'

def preprocess_live_frame(frame):
    """Preprocess the frame for deepfake detection."""
    if frame is None or frame.size == 0:
        print("Empty frame detected.")
        return None
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    return preprocessed_frame

# Ensure directories exist
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FEATURES_FOLDER, exist_ok=True)

# Load the trained deepfake detection model
model_static = tf.keras.models.load_model(MODEL_PATH_STATIC)
model_live = tf.keras.models.load_model(MODEL_PATH_LIVE)

# Centroid Tracker for multi-person tracking
# This class tracks multiple objects based on their centroids and maintains a history of keypoints
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0   # next free id for new objects
        self.objects = {}  # id: centroid
        self.keypoints_history = {}  # id: [list of keypoints per frame]
        self.max_distance = max_distance

    def update(self, detections): # detections: list of (centroid, keypoints)

        # If no objects are being tracked, initialize objects with detections
        if len(self.objects) == 0:
            for centroid, keypoints in detections: 
                self.objects[self.next_id] = centroid 
                self.keypoints_history[self.next_id] = [keypoints] 
                self.next_id += 1 
            return self.objects, self.keypoints_history

        # Else match detections to existing objects
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values())) # get centroids of existing objects
        detection_centroids = np.array([d[0] for d in detections]) # get centroids of new detections

        # If no detections or objects, return current state
        if len(object_centroids) == 0 or len(detection_centroids) == 0:
            return self.objects, self.keypoints_history

        # Compute distance matrix and find the best matches
        D = distance.cdist(object_centroids, detection_centroids) 
        rows = D.min(axis=1).argsort() # sort rows by minimum distance
        cols = D.argmin(axis=1)[rows] # get the column indices of the minimum distances


        # Assign detections to existing objects based on distance
        assigned = set() # track assigned detections
        for row, col in zip(rows, cols):
            if D[row, col] > self.max_distance or col in assigned:
                continue
            object_id = object_ids[row]
            centroid, keypoints = detections[col]
            self.objects[object_id] = centroid
            self.keypoints_history[object_id].append(keypoints)
            assigned.add(col)

        # Add new detections
        for i, (centroid, keypoints) in enumerate(detections):
            if i not in assigned:
                self.objects[self.next_id] = centroid
                self.keypoints_history[self.next_id] = [keypoints]
                self.next_id += 1

        return self.objects, self.keypoints_history
    
# === Step 0.5: Preprocess Video Frames ===
def preprocess_frame(frame):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Gamma correction
    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, look_up_table)

    # White balance correction (OpenCV’s xphoto module)
    wb = cv2.xphoto.createSimpleWB()
    white_balanced = wb.balanceWhite(gamma_corrected)

    return white_balanced



# === Step 1: Extract and Track Keypoints from Video ===
def extract_keypoints(video_path, show_process=True):
    predictor = openpifpaf.Predictor(checkpoint='resnet50')
    cap = cv2.VideoCapture(video_path)
    tracker = CentroidTracker()
    frame_num = 0

    while cap.isOpened():
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame = preprocess_frame(frame)

        # Get keypoints
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB for OpenPifPaf
        predictions, _, _ = predictor.numpy_image(rgb_frame) # get only predictions
        
        # Process predictions
        detections = []
        for ann in predictions:
            keypoints = []
            xs, ys = [], [] # lists to store x and y coordinates for centroid calculation
            for x, y, v in ann.data:
                keypoints.append((x, y, v))
                xs.append(x)
                ys.append(y)
    
            # Use mean of visible keypoints as centroid
            visible = [i for i in range(len(keypoints)) if keypoints[i][2] > 0.5]
            if visible:
                cx = np.mean([keypoints[i][0] for i in visible])
                cy = np.mean([keypoints[i][1] for i in visible])
            else:
                cx, cy = np.mean(xs), np.mean(ys)
            detections.append(((cx, cy), keypoints)) # centroid, keypoints

        # Update tracker with new detections
        objects, keypoints_history = tracker.update(detections)

        if show_process:
            for _, keypoints in detections:
                for i, (x, y, v) in enumerate(keypoints):
                    if v > 0.5:  # Only plot visible keypoints
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Optionally draw limbs (connect keypoints)
                # Example for some COCO keypoint connections:
                connections = [
                    (5, 7), (7, 9),     # Right Arm
                    (6, 8), (8, 10),    # Left Arm
                    (11, 13), (13, 15), # Right Leg
                    (12, 14), (14, 16), # Left Leg
                    (5, 6), (11, 12),   # Shoulders and Hips
                ]
                for idx1, idx2 in connections:
                    if keypoints[idx1][2] > 0.5 and keypoints[idx2][2] > 0.5:
                        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

            cv2.imshow("OpenPifPaf Pose Annotation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1

    cap.release()
    if show_process:
        cv2.destroyAllWindows()
    return keypoints_history

# === Step 2: Normalize Keypoints (Center on pelvis, scale by torso length) ===
def normalize_keypoints(keypoints_seq): # keypoints_seq: list of [ (x, y, v), ... ] per frame
    # Normalize each frame: center on pelvis (midpoint of hips), scale by torso length (shoulder to hip)
    norm_seq = []
    for frame in keypoints_seq:
        kp = np.array(frame) # get keypoints as numpy array
        if kp.shape[0] < 8:  # Not enough keypoints
            norm_seq.append(kp)
            continue

        # Math to find torso length
        # COCO keypoint format: 11=left hip, 12=right hip, 5=left shoulder, 6=right shoulder
        pelvis = np.mean([kp[11][:2], kp[12][:2]], axis=0)
        shoulder = np.mean([kp[5][:2], kp[6][:2]], axis=0)
        torso_length = np.linalg.norm(shoulder - pelvis) # get distance between shoulders and pelvis
        if torso_length < 1e-3: # avoid division by zero
            torso_length = 1.0
        normed = kp.copy()
        normed[:,0:2] = (kp[:,0:2] - pelvis) / torso_length # subtract pelvis position to normalise, divide by torso length to scale
        norm_seq.append(normed)
    return norm_seq

# == Step 3: Predict Deepfake using Sequence ==
def predict_deepfake(features):
    features_array = features.reshape(1, -1, 1)
    prediction = model_static.predict(features_array)
    confidence = float(prediction[0])
    threshold = 0.39
    result = "Fake" if confidence >= threshold else "Real"
    return result, confidence

# === Full Video Processing Pipeline ===
def detect_body_posture(video_path):
    """Process the video, extract keypoints, preprocess them, extract features, and predict deepfake."""
    
    # Step 1: Extract Keypoints
    keypoints_arr = extract_keypoints(video_path)
    if keypoints_arr is None:
        return {"error": "Failed to extract keypoints"}

    # Step 2: Preprocess Extracted Keypoints    
    normalized_arr = {}
    for person_id, seq in keypoints_arr.items():
        norm_seq = normalize_keypoints(seq)
        normalized_arr[person_id] = norm_seq 
    print("normalized_arr", normalized_arr)
    if normalized_arr is None:
        return {"error": "Failed to normalize keypoints"}

    # Step 3: Run Prediction
    result, confidence = predict_deepfake(normalized_arr)

    return {
        "prediction": result,
        "confidence": confidence,
    }

def preprocess_pose_input(keypoints):
    return np.array(keypoints).reshape(1, -1)  # Reshape to (1, 34)

def body_posture_live_detection():
    cap = cv2.VideoCapture(0)
    predictor = openpifpaf.Predictor(checkpoint='resnet50')

    real_frame_count = 0
    fake_frame_count = 0
    score_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🛑 No frame read from camera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions, _, _ = predictor.numpy_image(rgb)

        for ann in predictions:
            keypoints = [(x, y, v) for x, y, v in ann.data]
            visible = [i for i, (_, _, v) in enumerate(keypoints) if v > 0.5]

            # Draw keypoints
            for i in visible:
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

            # Draw limbs (example pairs)
            pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
            for i1, i2 in pairs:
                if keypoints[i1][2] > 0.5 and keypoints[i2][2] > 0.5:
                    pt1 = (int(keypoints[i1][0]), int(keypoints[i1][1]))
                    pt2 = (int(keypoints[i2][0]), int(keypoints[i2][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            # Deepfake prediction
            flat_pose = [coord for x, y, v in keypoints for coord in (x, y)]
            input_pose = preprocess_pose_input(flat_pose)
            score = model_live.predict(input_pose)[0][0]
            score_list.append(score)

            label = "Fake" if score > 0.5 else "Real"
            color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
            if label == "Fake":
                fake_frame_count += 1
            else:
                real_frame_count += 1

            loss_rate = 1 - score
            text = f'{score:.2f}, {loss_rate:.2f}, {label}'
            cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Live Pose-Based Deepfake Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def plot_and_save_graphs(score_list, real_count, fake_count, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    # Plot "Frames (Real vs Fake)"
    plt.figure(figsize=(8, 3))
    plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    plt.title('Frames (Real vs Fake)')
    plt.xlabel('Frame Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'frames_real_vs_fake.png'))
    plt.clf()

    # Plot "Confidence Score Over Time"
    plt.figure(figsize=(8, 3))
    plt.plot(score_list, color='blue')
    plt.title('Confidence Score Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Confidence Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confidence_score.png'))
    plt.clf()


def process_video_for_temporal_model(video_path, show_process=False):
    print("Extracting and tracking keypoints...")
    keypoints_history = extract_keypoints(video_path, show_process=show_process)
    for person_id, seq in keypoints_history.items():
        print(f"Processing person {person_id}...")
        norm_seq = normalize_keypoints(seq)
        # Save the full sequence for this person
        np.save(os.path.join(PROCESSED_FOLDER, f"person_{person_id}_sequence.npy"), norm_seq)
        print(f"Saved sequence for person {person_id}: shape {norm_seq.shape}")
