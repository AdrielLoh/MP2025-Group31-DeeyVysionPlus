import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from sklearn.preprocessing import normalize
import logging
import time
import matplotlib.pyplot as plt
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Caffe model for face detection
net = cv2.dnn.readNetFromCaffe('models/weights-prototxt.txt', 'models/res_ssd_300Dim.caffeModel')

# Initialize Mediapipe for face landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Load the trained model
model = tf.keras.models.load_model('models/visual_artifacts.keras')

# Global variables for metrics
frame_indices = []
confidence_scores = []
landmark_norms = []
dnn_feature_norms = []
detection_times = []
predictions = []
frame_labels = []

def extract_features(image, net, fixed_length=4096):
    try:
        start_time = time.time()
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection_time = time.time() - start_time
        
        if not results.multi_face_landmarks:
            return None, None, detection_time, None, None

        landmarks = results.multi_face_landmarks[0].landmark
        landmark_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        
        # Normalize landmark coordinates
        landmark_coords = normalize(landmark_coords.reshape(1, -1)).flatten()
        landmark_norm = np.linalg.norm(landmark_coords)
        
        # DNN Features
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        dnn_features = net.forward().flatten()
        dnn_feature_norm = np.linalg.norm(dnn_features)

        # Combine all features
        combined_features = np.hstack((landmark_coords, dnn_features))

        # Ensure the feature vector has the fixed length
        if len(combined_features) > fixed_length:
            combined_features = combined_features[:fixed_length]
        elif len(combined_features) < fixed_length:
            combined_features = np.pad(combined_features, (0, fixed_length - len(combined_features)), 'constant')

        return combined_features, landmarks, detection_time, landmark_norm, dnn_feature_norm

    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None, None, None, None, None


def extract_features_static(image, net, fixed_length=4096):
    """Extracts facial landmarks and DNN features from a given image."""
    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return np.zeros((fixed_length,), dtype=np.float32)  # Return zero-filled vector instead of None

        landmarks = results.multi_face_landmarks[0].landmark
        landmark_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        landmark_coords = normalize(landmark_coords.reshape(1, -1)).flatten()

        # DNN Features
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        dnn_features = net.forward().flatten()

        # Combine features
        combined_features = np.hstack((landmark_coords, dnn_features))

        # Ensure the feature vector has the fixed length
        if len(combined_features) > fixed_length:
            combined_features = combined_features[:fixed_length]
        elif len(combined_features) < fixed_length:
            combined_features = np.pad(combined_features, (0, fixed_length - len(combined_features)), 'constant')

        return combined_features  # Return only the feature vector
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return np.zeros((fixed_length,), dtype=np.float32)  # Return a zero vector instead of None


def draw_face_landmarks(image, landmarks):
    h, w, _ = image.shape
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

def save_graphs(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not frame_indices:
        logging.error("No frames were processed. Cannot generate graphs.")
        return

    # Confidence Over Time
    plt.figure()
    valid_confidence = [score for score in confidence_scores if score is not None]
    valid_frames = [frame for frame, score in zip(frame_indices, confidence_scores) if score is not None]
    if valid_confidence:
        plt.plot(valid_frames, valid_confidence, label='Prediction Confidence')
        plt.xlabel('Frame Index')
        plt.ylabel('Confidence')
        plt.title('Prediction Confidence Over Time')
        plt.savefig(os.path.join(output_folder, 'confidence_over_time.png'))
    else:
        logging.warning("No valid confidence scores to plot.")
    plt.close()

    # Landmark Norms Over Time
    plt.figure()
    valid_norms = [norm for norm in landmark_norms if norm is not None]
    if valid_norms:
        plt.plot(frame_indices[:len(valid_norms)], valid_norms, label='Landmark Norms')
        plt.xlabel('Frame Index')
        plt.ylabel('Norm')
        plt.title('Landmark Norms Over Time')
        plt.savefig(os.path.join(output_folder, 'landmark_norms_over_time.png'))
    else:
        logging.warning("No valid landmark norms to plot.")
    plt.close()

    # DNN Feature Norms Over Time
    plt.figure()
    valid_dnn_norms = [norm for norm in dnn_feature_norms if norm is not None]
    if valid_dnn_norms:
        plt.plot(frame_indices[:len(valid_dnn_norms)], valid_dnn_norms, label='DNN Feature Norms')
        plt.xlabel('Frame Index')
        plt.ylabel('Norm')
        plt.title('DNN Feature Norms Over Time')
        plt.savefig(os.path.join(output_folder, 'dnn_feature_norms_over_time.png'))
    else:
        logging.warning("No valid DNN feature norms to plot.")
    plt.close()

    # Face Detection Rate
    plt.figure()
    valid_times = [time for time in detection_times if time is not None]
    if valid_times:
        plt.plot(frame_indices[:len(valid_times)], valid_times, label='Face Detection Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Time (s)')
        plt.title('Face Detection Time Over Time')
        plt.savefig(os.path.join(output_folder, 'detection_times_over_time.png'))
    else:
        logging.warning("No valid detection times to plot.")
    plt.close()

    # Real vs. Fake Frames Bar Graph
    plt.figure()
    real_count = frame_labels.count(1)
    fake_count = frame_labels.count(0)
    if real_count + fake_count > 0:
        plt.bar(['Fake', 'Real'], [fake_count, real_count], color=['red', 'green'], alpha=0.7)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Real vs Fake Frame Counts')
        plt.savefig(os.path.join(output_folder, 'real_vs_fake.png'))
    else:
        logging.warning("No frames were classified as real or fake.")
    plt.close()

def live_detection(output_folder):
    global frame_indices, confidence_scores, landmark_norms, dnn_feature_norms, detection_times, predictions, frame_labels

    frame_indices = []
    confidence_scores = []
    landmark_norms = []
    dnn_feature_norms = []
    detection_times = []
    predictions = []
    frame_labels = []

    cap = cv2.VideoCapture(1)  # Change to 1 or the appropriate camera index if needed
    frame_index = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture image.")
                break
            
            try:
                result = extract_features_static(frame, net)
                
                if result is None or len(result) != 5:
                    cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Live Detection', frame)
                    detection_times.append(None)
                    landmark_norms.append(None)
                    dnn_feature_norms.append(None)
                    confidence_scores.append(None)
                    frame_indices.append(frame_index)
                    predictions.append("No Face")
                    frame_labels.append(0)
                else:
                    features, landmarks, detection_time, landmark_norm, dnn_feature_norm = result
                    
                    detection_times.append(detection_time)
                    landmark_norms.append(landmark_norm)
                    dnn_feature_norms.append(dnn_feature_norm)
                    frame_indices.append(frame_index)
                    
                    features = np.expand_dims(features, axis=0)
                    
                    prediction = model.predict(features)[0][0]
                    confidence_scores.append(prediction)
                    
                    label = 'Fake' if prediction < 0.8 else 'Real'
                    predictions.append(label)
                    frame_labels.append(1 if label == 'Real' else 0)
                    
                    label_text = f"{label}: {prediction:.2f}"
                    color = (0, 255, 0) if label == 'Real' else (0, 0, 255)
                    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    if landmarks:
                        draw_face_landmarks(frame, landmarks)
                
                cv2.imshow('Live Detection', frame)
                
                if frame_index % 100 == 0:
                    logging.info(f"Processed {frame_index} frames. "
                                 f"Confidence scores: {len(confidence_scores)}, "
                                 f"Landmark norms: {len(landmark_norms)}, "
                                 f"Frame labels: {len(frame_labels)}")
                
                frame_index += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                continue
        
    except Exception as e:
        logging.error(f"Error in live detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    logging.info(f"Total frames processed: {frame_index}")
    logging.info(f"Confidence scores: {len(confidence_scores)}")
    logging.info(f"Landmark norms: {len(landmark_norms)}")
    logging.info(f"Frame labels: {len(frame_labels)}")

    save_graphs(output_folder)
    
    total_frames = len(frame_labels)
    real_count = sum(frame_labels)
    fake_count = total_frames - real_count
    overall_result = "Real" if real_count > fake_count else "Fake"
    
    return overall_result, real_count, fake_count

def static_video_detection(video_path, output_folder):
    """Performs deepfake detection on an uploaded video file."""
    cap = cv2.VideoCapture(video_path)
    frame_indices.clear()
    confidence_scores.clear()
    frame_labels.clear()
    frame_index = 0
    real_frame_count = 0
    fake_frame_count = 0

    if not cap.isOpened():
        logging.error("Error: Unable to open video file.")
        return "Error", 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_features_static(frame, net)

        # Ensure features is a NumPy array
        if not isinstance(features, np.ndarray):
            print("Error: Extracted features are not a valid NumPy array.")
            continue

        features = features.reshape(1, -1).astype(np.float32)  # Reshape correctly

        # Debugging: Print shape before prediction
        print(f"Features shape before prediction: {features.shape}")

        try:
            prediction = model.predict(features)[0][0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error", 0, 0

        confidence_scores.append(prediction)

        if prediction > 0.38:
            fake_frame_count += 1
            frame_labels.append(0)  # Fake
        else:
            real_frame_count += 1
            frame_labels.append(1)  # Real
        
        frame_indices.append(frame_index)
        frame_index += 1

    cap.release()
    save_graphs(output_folder)

    overall_result = "Real" if real_frame_count > fake_frame_count else "Fake"
    return overall_result, real_frame_count, fake_frame_count


if __name__ == "__main__":
    output_folder = 'uploads/results'
    overall_result, real_count, fake_count = live_detection(output_folder)
    print(f"Overall result: {overall_result}")
    print(f"Real frames: {real_count}")
    print(f"Fake frames: {fake_count}")