# Save this entire file as: detection_scripts/visual_artifacts_script.py
import cv2
import numpy as np
import tensorflow as tf
import os
import logging
import time
import matplotlib
# Use a non-interactive backend to prevent GUI errors on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MODEL LOADING ---
FACE_DETECTOR = None
DEEPFAKE_DETECTOR = None
try:
    FACE_DETECTOR = cv2.dnn.readNetFromCaffe('models/weights-prototxt.txt', 'models/res_ssd_300Dim.caffeModel')
    logging.info("Caffe Face Detector model loaded successfully for visual artifacts script.")
    model_path = 'models/better_visual_artifacts.keras'
    DEEPFAKE_DETECTOR = tf.keras.models.load_model(model_path)
    logging.info(f"Visual Artifacts Detector model loaded successfully from: {model_path}")
except Exception as e:
    logging.error(f"Fatal: Could not load a required model. Error: {e}")

def _process_frame_and_predict(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    FACE_DETECTOR.setInput(blob)
    detections = FACE_DETECTOR.forward()
    
    if detections.shape[2] > 0:
        best_detection_index = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, best_detection_index, 2]
        if confidence > 0.5:
            box = detections[0, 0, best_detection_index, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            if face.size == 0: return None
            face_resized = cv2.resize(face, (380, 380))
            face_normalized = face_resized.astype("float32") / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            prediction = DEEPFAKE_DETECTOR.predict(face_batch, verbose=0)[0][0]
            return prediction
    return None

def save_analysis_graphs(output_folder, frame_indices, confidence_scores, detection_times, real_count, fake_count):
    """
    Plots and saves graphs of relevant metrics from the detection process.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if not frame_indices:
        logging.warning("No frames were processed. Cannot generate graphs.")
        return

    # Plot 1: Prediction Confidence Over Processed Frames
    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, confidence_scores, marker='o', linestyle='-', label='Prediction Confidence')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')
    plt.xlabel('Frame Number')
    plt.ylabel('Confidence (Probability of Fake)')
    plt.title('Prediction Confidence Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'visual_artifacts_confidence.png'))
    plt.close()

    # Plot 2: Detection Time per Processed Frame
    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, detection_times, marker='o', linestyle='-', color='orange', label='Processing Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Time (seconds)')
    plt.title('Processing Time per Sampled Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'visual_artifacts_times.png'))
    plt.close()

    # Plot 3: Final Counts
    plt.figure(figsize=(8, 6))
    plt.bar(['Real Frames', 'Fake Frames'], [real_count, fake_count], color=['green', 'red'])
    plt.ylabel('Count')
    plt.title('Final Frame Classification Counts')
    plt.savefig(os.path.join(output_folder, 'visual_artifacts_counts.png'))
    plt.close()
    
    logging.info(f"Analysis graphs saved to {output_folder}")

def live_detection(output_folder):
    if not FACE_DETECTOR or not DEEPFAKE_DETECTOR:
        raise ConnectionError("Models not loaded.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return "Error: Webcam not found", 0, 0

    logging.info("Starting visual artifacts live detection...")
    real_count, fake_count = 0, 0
    start_time = time.time()
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret: break
        
        prediction = _process_frame_and_predict(frame)
        if prediction is not None:
            if prediction > 0.5: fake_count += 1
            else: real_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    if real_count == 0 and fake_count == 0: return "No face detected", 0, 0
    overall_result = "FAKE" if fake_count > real_count else "REAL"
    return overall_result, real_count, fake_count

def static_video_detection(video_path, output_folder):
    """
    Processes a static video file using FRAME SAMPLING for speed.
    """
    if not FACE_DETECTOR or not DEEPFAKE_DETECTOR:
        raise ConnectionError("Models not loaded.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return "Error: Could not open video", 0, 0

    # --- FRAME SAMPLING LOGIC ---
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Process 2 frames per second. If FPS is low, process every frame.
    frame_skip_interval = max(1, int(video_fps / 2))
    logging.info(f"Video FPS: {video_fps:.2f}. Processing 1 frame every {frame_skip_interval} frames.")

    real_count, fake_count = 0, 0
    frame_num = 0
    
    # Lists to store data for plotting
    processed_frame_indices = []
    confidence_scores = []
    detection_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Only process frames at the specified interval
        if frame_num % frame_skip_interval == 0:
            proc_start_time = time.time()
            prediction = _process_frame_and_predict(frame)
            proc_end_time = time.time()
            
            if prediction is not None:
                # Store results for graphing
                processed_frame_indices.append(frame_num)
                confidence_scores.append(prediction)
                detection_times.append(proc_end_time - proc_start_time)
                
                if prediction > 0.5: fake_count += 1
                else: real_count += 1
        
        frame_num += 1
        
    cap.release()
    
    # Save the analysis graphs now that we have the data
    if processed_frame_indices:
        save_analysis_graphs(output_folder, processed_frame_indices, confidence_scores, detection_times, real_count, fake_count)
    
    if real_count == 0 and fake_count == 0: return "No face detected in video", 0, 0
    overall_result = "FAKE" if fake_count > real_count else "REAL"
    return overall_result, real_count, fake_count
