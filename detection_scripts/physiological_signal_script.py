import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import joblib
import os
from collections import deque
import time

# Load the trained model
clf = joblib.load('models/deepfake_detection_model.pkl')

# Load the face detection model
face_net = cv2.dnn.readNetFromCaffe('models/weights-prototxt.txt', 'models/res_ssd_300Dim.caffeModel')

# Butterworth filter to smooth the signal
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to calculate heart rate from signal peaks
def calculate_heart_rate(signal, fs):
    peaks, _ = find_peaks(signal, distance=fs/2)
    peak_intervals = np.diff(peaks) / fs
    if len(peak_intervals) > 0:
        heart_rate = 60.0 / np.mean(peak_intervals)
        return heart_rate
    return 0

# Function to calculate additional physiological features
def calculate_additional_features(signal, fs):
    sdnn = np.std(np.diff(signal))
    blood_flow_pattern = np.mean(signal)
    return sdnn, blood_flow_pattern

# Function to extract signals from multiple facial regions under green light
def extract_signals_from_regions(frame, faces):
    signals = []
    
    for (x, y, w, h) in faces:
        rois = [
            frame[y:y+h, x:x+w],
            frame[y:int(y+h/2), x:x+w],
            frame[int(y+h/2):y+h, x:x+w],
            frame[int(y+h*0.2):int(y+h*0.4), x:x+w],
            frame[int(y+h*0.6):int(y+h*0.8), x:x+w]
        ]
        
        for roi in rois:
            green_channel = roi[:, :, 1]  # Green channel for green light
            mean_green = np.mean(green_channel)
            signals.append(mean_green)
    
    return signals

# Function to check lighting condition
def check_lighting(frame, threshold=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > threshold

# Function to plot and save individual graphs
def plot_and_save_graphs(signals, filtered_signals, heart_rate, prediction, confidence, real_count, fake_count, output_folder):
    # Plot original signal
    plt.figure(figsize=(8, 3))
    plt.plot(signals, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'original_signal.png'))
    plt.clf()

    # Plot filtered signal
    plt.figure(figsize=(8, 3))
    plt.plot(filtered_signals, label='Filtered Signal', color='orange')
    plt.title('Filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'filtered_signal.png'))
    plt.clf()

    # Plot heart rate distribution
    plt.figure(figsize=(8, 3))
    plt.hist(filtered_signals, bins=50, color='green')
    plt.title(f'Heart Rate Distribution - {heart_rate:.2f} BPM')
    plt.xlabel('Heart Rate')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'heart_rate_distribution.png'))
    plt.clf()

    # Plot prediction result
    plt.figure(figsize=(8, 3))

    if prediction == 'REAL':
        result = 'REAL' 
        facecolor = 'green'
    elif prediction == 'FAKE':
        result = 'FAKE'
        facecolor = 'red'
    else:
        result = 'UNKNOWN'
        facecolor = 'yellow'

    plt.text(0.5, 0.5, f'Prediction: {result}\nConfidence: {confidence:.2f}', 
            horizontalalignment='center', verticalalignment='center', 
            fontsize=12, bbox=dict(facecolor=facecolor, alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'prediction_result.png'))
    plt.clf()

    # Plot counts of real and fake predictions
    plt.figure(figsize=(8, 3))
    plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    plt.title('Counts of Real and Fake Predictions')
    plt.xlabel('Prediction Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'prediction_counts.png'))
    plt.clf()

# Function to detect faces using deep learning model
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

# Main detection function for static files
def detect_physiological_signal(filename, output_folder, threshold=0.5):
    cap = cv2.VideoCapture(filename)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None, 0, 0, 0, 0

    fs = 30.0  # Frame rate
    signal_buffer = []
    real_count = 0
    fake_count = 0
    all_signals = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces_dnn(frame)
        
        frame_signals = extract_signals_from_regions(frame, faces)
        signal_buffer.extend(frame_signals)
        all_signals.extend(frame_signals)

        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Default green rectangle

        label_color = (0, 255, 0)  # Green by default
        label = "Processing..."
        confidence_text = ""

        if len(signal_buffer) >= fs * 5:  # 5 seconds worth of data
            signals = np.array(signal_buffer)
            filtered_signals = bandpass_filter(signals, lowcut=0.75, highcut=2.5, fs=fs, order=5)
            hr = calculate_heart_rate(filtered_signals, fs)
            sdnn, blood_flow_pattern = calculate_additional_features(filtered_signals, fs)
            
            # Prepare features for prediction
            features = [[hr, sdnn, blood_flow_pattern]]
            prediction_proba = clf.predict_proba(features)[0]
            prediction = np.argmax(prediction_proba)
            confidence = prediction_proba[prediction]

            # Apply threshold to determine real or fake
            if prediction == 1 and confidence >= threshold:
                fake_count += 1
                label = f'FAKE - HR: {hr:.2f} BPM'
                label_color = (0, 0, 255)  # Red
            else:
                real_count += 1
                label = f'REAL - HR: {hr:.2f} BPM'
                label_color = (0, 255, 0)  # Green
            
            confidence_text = f'Confidence: {confidence:.2f}'

        # Draw the prediction label and confidence on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2, cv2.LINE_AA)
        cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2, cv2.LINE_AA)

        # Display stopwatch
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f'Time: {int(elapsed_time)}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Static Deepfake Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Determine final conclusion based on the counts
    if real_count > fake_count:
        final_prediction = "REAL"
    else:
        final_prediction = "FAKE"

    # Process all collected signals for visualization
    if all_signals:
        signals = np.array(all_signals)
        filtered_signals = bandpass_filter(signals, lowcut=0.75, highcut=2.5, fs=fs, order=5)
        hr = calculate_heart_rate(filtered_signals, fs)

        if (real_count + fake_count) > 0:  # Prevent division by zero
            confidence = max(real_count, fake_count) / (real_count + fake_count)
        else:
            final_prediction = "UNKNOWN"
            confidence = 0  # or handle it appropriately based on your logic

        plot_and_save_graphs(signals, filtered_signals, hr, final_prediction, confidence, real_count, fake_count, output_folder)

    return final_prediction, hr, confidence, real_count, fake_count

# Real-time video capture and deepfake detection
def real_time_detection(output_folder, threshold=0.5):
    cap = cv2.VideoCapture(1)  # Use webcam; change to video file path if needed
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None, 0, 0

    fs = 30.0  # Frame rate
    signal_buffer = deque(maxlen=int(fs*10))  # Buffer for 10 seconds
    real_count = 0
    fake_count = 0
    start_time = time.time()
    max_duration = 15  # Maximum duration in seconds
    prediction = None
    hr = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check lighting condition
        if not check_lighting(frame):
            cv2.putText(frame, "Lighting is too dark!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        faces = detect_faces_dnn(frame)
        
        frame_signals = extract_signals_from_regions(frame, faces)
        signal_buffer.extend(frame_signals)
        
        # Draw rectangle around faces and display the frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Default green rectangle

        # Prepare prediction label (green/red) based on the detection
        label_color = (0, 255, 0)  # Green by default
        label = "Processing..."
        confidence_text = ""

        if len(signal_buffer) >= fs * 5:  # 5 seconds worth of data
            signals = np.array(signal_buffer)
            filtered_signals = bandpass_filter(signals, lowcut=0.75, highcut=2.5, fs=fs, order=5)
            hr = calculate_heart_rate(filtered_signals, fs)
            sdnn, blood_flow_pattern = calculate_additional_features(filtered_signals, fs)
            
            features = [[hr, sdnn, blood_flow_pattern]]
            prediction_proba = clf.predict_proba(features)[0]
            prediction = np.argmax(prediction_proba)
            confidence = prediction_proba[prediction]

            # Apply threshold to determine real or fake
            if prediction == 1 and confidence >= threshold:
                fake_count += 1
                label = f'FAKE - HR: {hr:.2f} BPM'
                label_color = (0, 0, 255)  # Red
            else:
                real_count += 1
                label = f'REAL - HR: {hr:.2f} BPM'
                label_color = (0, 255, 0)  # Green
            
            confidence_text = f'Confidence: {confidence:.2f}'

        # Draw the prediction label and confidence on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2, cv2.LINE_AA)
        cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2, cv2.LINE_AA)

        # Display stopwatch
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f'Time: {int(elapsed_time)}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Real-time Deepfake Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if elapsed_time >= max_duration:
            break

    cap.release()
    cv2.destroyAllWindows()

    if real_count > fake_count:
        final_prediction = "REAL"
    else:
        final_prediction = "FAKE"

    if len(signal_buffer) > 0:
        signals = np.array(signal_buffer)
        filtered_signals = bandpass_filter(signals, lowcut=0.75, highcut=2.5, fs=fs, order=5)
        hr = calculate_heart_rate(filtered_signals, fs)

        if (real_count + fake_count) > 0:  # Prevent division by zero
            confidence = max(real_count, fake_count) / (real_count + fake_count)
        else:
            final_prediction = "UNKNOWN"
            confidence = 0  # or handle it appropriately based on your logic

        plot_and_save_graphs(signals, filtered_signals, hr, final_prediction, confidence, real_count, fake_count, output_folder)

    return final_prediction, hr, confidence, real_count, fake_count