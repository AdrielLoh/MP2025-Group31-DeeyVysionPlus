import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import joblib
from collections import deque
import time

# Load the trained model
clf = joblib.load('Website/models/rppg_model_0.67_v2.pkl')

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

def calculate_heart_rate(signal, fs):
    peaks, _ = find_peaks(signal, distance=fs/2)
    peak_intervals = np.diff(peaks) / fs
    if len(peak_intervals) > 0:
        heart_rate = 60.0 / np.mean(peak_intervals)
        return heart_rate
    return 0

def calculate_additional_features(signal, fs):
    sdnn = np.std(np.diff(signal))
    blood_flow_pattern = np.mean(signal)
    return sdnn, blood_flow_pattern

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

def check_lighting(frame, threshold=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > threshold

def plot_graphs(signals, filtered_signals, heart_rate, prediction, confidence, real_count, fake_count):
    plt.figure(figsize=(15, 10))

    plt.subplot(5, 1, 1)
    plt.plot(signals, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal Intensity')
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(filtered_signals, label='Filtered Signal', color='orange')
    plt.title('Filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal Intensity')
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.hist(filtered_signals, bins=50, color='green')
    plt.title(f'Heart Rate Distribution - {heart_rate:.2f} BPM')
    plt.xlabel('Heart Rate')
    plt.ylabel('Frequency')

    plt.subplot(5, 1, 4)
    result = 'REAL' if prediction == 0 else 'FAKE'
    plt.text(0.5, 0.5, f'Prediction: {result}\nConfidence: {confidence:.2f}', 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=12, bbox=dict(facecolor='red' if result == 'FAKE' else 'green', alpha=0.5))
    plt.axis('off')

    plt.subplot(5, 1, 5)
    plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    plt.title('Counts of Real and Fake Predictions')
    plt.xlabel('Prediction Type')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def real_time_detection():
    cap = cv2.VideoCapture(0)  # Use webcam; change to video file path if needed
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    fs = 30.0  # Frame rate
    signal_buffer = deque(maxlen=int(fs*10))  # Buffer for 10 seconds
    prediction = None
    hr = None
    start_time = time.time()
    min_duration = 30  # Minimum duration in seconds
    real_count = 0
    fake_count = 0
    stop_warning = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check lighting condition
        if not check_lighting(frame):
            cv2.putText(frame, "Lighting is too dark!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        frame_signals = extract_signals_from_regions(frame, faces)
        signal_buffer.extend(frame_signals)
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display stopwatch
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f'Time: {int(elapsed_time)}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Prepare features for prediction if enough signals are collected
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

            # Count predictions
            if prediction == 0:
                real_count += 1
            else:
                fake_count += 1

            # Display prediction and heart rate on the frame
            if prediction == 0:
                label = f'REAL - HR: {hr:.2f} BPM'
                color = (0, 255, 0)  # Green
            elif prediction == 1:
                label = f'FAKE - HR: {hr:.2f} BPM'
                color = (0, 0, 255)  # Red
            
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('Real-time Deepfake Detection', frame)
        
        # Break the loop on 'q' key press if minimum duration is met
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if elapsed_time >= min_duration:
                break
            else:
                stop_warning = True
        
        # Show warning if user attempts to stop before 30 seconds
        if stop_warning:
            cv2.putText(frame, f"Please wait for at least {min_duration - int(elapsed_time)} more seconds.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Real-time Deepfake Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for user to release 'q' key
                stop_warning = False

    cap.release()
    cv2.destroyAllWindows()

    # Plot graphs when the process is stopped
    if len(signal_buffer) > 0:
        signals = np.array(signal_buffer)
        filtered_signals = bandpass_filter(signals, lowcut=0.75, highcut=2.5, fs=fs, order=5)
        hr = calculate_heart_rate(filtered_signals, fs)
        plot_graphs(signals, filtered_signals, hr, prediction, confidence, real_count, fake_count)

    # Return the prediction and heart rate
    if prediction is not None and hr is not None:
        return prediction, hr
    else:
        return None, None

# Ensure this block only runs when the script is executed directly
if __name__ == "__main__":
    # Run real-time detection
    prediction, hr = real_time_detection()
    if prediction is not None:
        if prediction == 0:
            print(f'The video is predicted to be REAL with an average heart rate of {hr:.2f} BPM.')
        elif prediction == 1:
            print(f'The video is predicted to be FAKE with an average heart rate of {hr:.2f} BPM.')
    else:
        print('Error: Could not detect any faces or extract signals.')
