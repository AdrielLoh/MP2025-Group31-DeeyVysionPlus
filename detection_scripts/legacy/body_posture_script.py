import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt


# Define paths for storing processed files
PROCESSED_FOLDER = "static/processed/"
FEATURES_FOLDER = "static/features/"
MODEL_PATH_STATIC = 'models/body_posture.keras'  
MODEL_PATH_LIVE = 'models/body_posture_live.keras'


# Load the face detection model
face_net = cv2.dnn.readNetFromCaffe('models/weights-prototxt.txt', 'models/res_ssd_300Dim.caffeModel')

os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FEATURES_FOLDER, exist_ok=True)

# Load the trained deepfake detection model
model_static = tf.keras.models.load_model(MODEL_PATH_STATIC)
model_live = tf.keras.models.load_model(MODEL_PATH_LIVE)

# === Step 1: Extract Keypoints from Video ===
def extract_keypoints(video_path):
    """Extract keypoints from a video using MediaPipe Pose estimation."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            keypoints = [
                (lm.x, lm.y, lm.z, lm.visibility)
                for lm in results.pose_landmarks.landmark
            ]
            keypoints_list.append(keypoints)

    cap.release()
    pose.close()

    return keypoints_list

def extract_keypoints_from_video(video_path):
    """Extract keypoints from a video and save as CSV."""
    keypoints_csv = os.path.join(PROCESSED_FOLDER, os.path.basename(video_path).replace('.mp4', '_keypoints.csv'))
    keypoints_list = extract_keypoints(video_path)

    if not keypoints_list:
        return None

    # Flatten keypoints and save to CSV
    flattened_keypoints = [keypoint for frame in keypoints_list for keypoint in frame]
    df = pd.DataFrame(flattened_keypoints)
    df.to_csv(keypoints_csv, index=False, header=False)

    return keypoints_csv

# === Step 2: Preprocess Keypoints Data ===
def preprocess_csv(input_csv):
    """Preprocess extracted keypoints data by normalizing features."""
    preprocessed_csv = os.path.join(PROCESSED_FOLDER, os.path.basename(input_csv).replace('_keypoints.csv', '_preprocessed.csv'))
    
    try:
        df = pd.read_csv(input_csv, header=None)

        if df.empty:
            return None

        # Normalize the features
        features = df.iloc[:, :-1]  # All columns except the last
        labels = df.iloc[:, -1]  # Last column (visibility)

        features_normalized = (features - features.min()) / (features.max() - features.min())
        features_normalized = features_normalized.fillna(0)

        processed_df = pd.concat([features_normalized, labels], axis=1)
        processed_df.to_csv(preprocessed_csv, index=False, header=False)

        return preprocessed_csv
    except Exception as e:
        print(f"Error preprocessing keypoints: {e}")
        return None

# === Step 3: Extract Statistical Features ===
def extract_features(input_csv):
    """Extract statistical features from preprocessed keypoints."""
    features_csv = os.path.join(FEATURES_FOLDER, os.path.basename(input_csv).replace('_preprocessed.csv', '_features.csv'))
    
    try:
        data = pd.read_csv(input_csv, header=None)

        features = {
            'mean_x': data.iloc[:, 0].mean(),
            'std_x': data.iloc[:, 0].std(),
            'mean_y': data.iloc[:, 1].mean(),
            'std_y': data.iloc[:, 1].std(),
            'mean_z': data.iloc[:, 2].mean(),
            'std_z': data.iloc[:, 2].std(),
            'max_confidence': data.iloc[:, 3].max(),
            'min_confidence': data.iloc[:, 3].min(),
        }

        features_df = pd.DataFrame([features])
        features_df.to_csv(features_csv, index=False)

        return features_csv
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# === Step 4: Run Extracted Features Through Model ===
def predict_deepfake(features_csv):
    """Load extracted features and run through deepfake detection model."""
    try:
        df = pd.read_csv(features_csv)
        features_array = df.values  # Convert DataFrame to NumPy array

        # Ensure correct shape (batch_size, sequence_length=8, channels=1)
        features_array = features_array.reshape(1, 8, 1)

        # Predict using the model
        prediction = model_static.predict(features_array)
        confidence = float(prediction[0])  # Convert tensor output to float


        # Set classification threshold
        threshold = 0.39 

        if confidence >= threshold:
            result = "Fake"
        else:
            result = "Real"

        print(f"Prediction Confidence Score: {result}")

        return result, confidence
    except Exception as e:
        print(f"Error running model prediction: {e}")
        return "Error", None

    
# === Full Video Processing Pipeline ===
def detect_body_posture(video_path, video_tag=False):
    """Process the video, extract keypoints, preprocess them, extract features, and predict deepfake."""
    
    # Step 1: Extract Keypoints
    keypoints_csv = extract_keypoints_from_video(video_path)
    if keypoints_csv is None:
        return {"error": "Failed to extract keypoints"}

    # Step 2: Preprocess Extracted Keypoints
    preprocessed_csv = preprocess_csv(keypoints_csv)
    if preprocessed_csv is None:
        return {"error": "Failed to preprocess keypoints"}

    # Step 3: Extract Features
    features_csv = extract_features(preprocessed_csv)
    if features_csv is None:
        return {"error": "Failed to extract features"}

    # Step 4: Run Prediction
    result, confidence = predict_deepfake(features_csv)

    return {
        "prediction": result,
        "confidence": confidence,
    }


def detect_faces_dnn(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold to filter weak detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

def preprocess_frame(frame):
    if frame.size == 0:
        print("Warning: Trying to resize an empty frame.")
        return None
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

def body_posture_live_detection(output_folder):
    cap = cv2.VideoCapture(2)  # 0 for default camera
    real_frame_count = 0
    fake_frame_count = 0
    score_list = []  # To store confidence scores

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame read from camera.")
            break

        faces = detect_faces_dnn(frame)
        if not faces:
            print("No faces detected.")
            continue

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                print(f"Empty face region detected: x={x}, y={y}, w={w}, h={h}")
                continue

            preprocessed_face = preprocess_frame(face)
            if preprocessed_face is None:
                continue

            prediction = model_live.predict(preprocessed_face)
            score = prediction[0][0]
            score_list.append(score)

            if score > 0.5:
                label = 'Fake'
                fake_frame_count += 1
                color = (0, 0, 255)
            else:
                label = 'Real'
                real_frame_count += 1
                color = (0, 255, 0)

            loss_rate = 1 - score
            text = f'{score:.2f}, {loss_rate:.2f}, {label}'
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Real-time Deepfake Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Determine overall result
    if fake_frame_count > real_frame_count:
        overall_result = "Fake"
    else:
        overall_result = "Real"

    # Generate and save graphs
    plot_and_save_graphs(score_list, real_frame_count, fake_frame_count, output_folder)

    return overall_result, real_frame_count, fake_frame_count


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
