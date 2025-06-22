import os
import cv2
import numpy as np
import joblib
from scipy.signal import detrend, butter, filtfilt, periodogram
import matplotlib.pyplot as plt
import uuid

# Load XGBoost model and feature scaler
clf = joblib.load('models/deepfake_detection_xgboost_best.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'
# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

def detect_faces_dnn(frame, conf_threshold=0.5):
    # Detect faces in a frame using OpenCV DNN
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def iou(boxA, boxB):
    # Intersection-over-union for face tracking
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def track_faces(prev_tracks, curr_boxes, frame_idx, iou_thresh=0.5):
    # Simple tracker: assign detected boxes to existing tracks using IoU
    assigned = [False] * len(curr_boxes)
    updated_tracks = []
    for track in prev_tracks:
        last_box = track['last_box']
        found = False
        for i, box in enumerate(curr_boxes):
            if not assigned[i] and iou(last_box, box) > iou_thresh:
                track['boxes'].append((frame_idx, box))
                track['last_box'] = box
                track['last_frame'] = frame_idx
                assigned[i] = True
                found = True
                break
        if found:
            updated_tracks.append(track)
    # Add new tracks for boxes not assigned
    for i, box in enumerate(curr_boxes):
        if not assigned[i]:
            updated_tracks.append({
                'id': len(updated_tracks),
                'boxes': [(frame_idx, box)],
                'last_box': box,
                'last_frame': frame_idx
            })
    return updated_tracks

def butter_bandpass_filter(signal, fs, lowcut=0.7, highcut=4.0, order=3):
    # Bandpass filter for rPPG extraction
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def plot_rppg_analysis(rppg_sig, f, pxx, save_dir, track_id):
    # Plot and save time-series rPPG and frequency spectrum for each face track
    os.makedirs(save_dir, exist_ok=True)
    plot_id = f"face_{track_id}_{uuid.uuid4().hex[:8]}"

    plt.figure()
    plt.plot(rppg_sig, label='rPPG Signal')
    plt.title(f'Face {track_id} - rPPG Time Series')
    plt.xlabel('Frame Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rppg_signal_{plot_id}.png'))
    plt.close()

    plt.figure()
    plt.plot(f, pxx, label='Power Spectrum')
    plt.title(f'Face {track_id} - rPPG Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rppg_spectrum_{plot_id}.png'))
    plt.close()

def compute_rppg_features(rgb_signal, fs):
    # rPPG extraction pipeline for a given RGB mean signal
    rgb_detrend = detrend(rgb_signal, axis=0)
    rgb_norm = (rgb_detrend - np.mean(rgb_detrend, axis=0)) / (np.std(rgb_detrend, axis=0) + 1e-8)
    S = rgb_norm
    h = (S[:, 1] - S[:, 0]) / (np.std(S[:, 1] - S[:, 0]) + 1e-8)
    s = (S[:, 1] + S[:, 0] - 2 * S[:, 2]) / (np.std(S[:, 1] + S[:, 0] - 2 * S[:, 2]) + 1e-8)
    rppg_sig = h + s
    rppg_sig = butter_bandpass_filter(rppg_sig, fs)
    f, pxx = periodogram(rppg_sig, fs)
    valid = (f >= 0.7) & (f <= 4.0)
    f, pxx = f[valid], pxx[valid]
    if len(f) == 0:
        # If invalid, return zero-filled outputs
        return [0]*9, 0, np.zeros_like(rppg_sig), np.zeros_like(f), np.zeros_like(pxx)
    peak_idx = np.argmax(pxx)
    hr_freq, hr_bpm, hr_power = f[peak_idx], f[peak_idx]*60, pxx[peak_idx]
    snr = hr_power / (np.sum(pxx) - hr_power + 1e-8)
    features = [np.mean(rppg_sig), np.std(rppg_sig), np.max(rppg_sig), np.min(rppg_sig),
                np.ptp(rppg_sig), hr_freq, hr_bpm, hr_power, snr]
    return features, hr_bpm, rppg_sig, f, pxx

def draw_output(frame, face_box, prediction, confidence, hr_bpm, time_stamp):
    # Draw bounding box, prediction label, confidence, and HR on output frame
    x, y, w, h = face_box
    label = f"{prediction.upper()} - HR: {hr_bpm:.2f} BPM"
    conf_label = f"Confidence: {confidence:.2f}"
    time_label = f"Time: {time_stamp}s"
    color = (0, 255, 0) if prediction == 'REAL' else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, conf_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, time_label, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def clamp_box(box, frame_shape):
    # Clamp face box coordinates to image bounds
    x, y, w, h = box
    H, W = frame_shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)

def run_detection(source, output_path='results/output.avi', is_webcam=False):
    # Main production pipeline for rPPG-based deepfake detection
    # Saves overlayed video and plots rPPG for each tracked face at end
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(0 if is_webcam else source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    all_predictions, real_count, fake_count, hr_values = [], 0, 0, []
    tracks = []
    face_rgb_history = {}  # {track_id: [ [r,g,b], ... ] }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Detect faces in current frame
        boxes = detect_faces_dnn(frame)
        # Track faces over time using IoU
        tracks = track_faces(tracks, boxes, frame_idx)
        for track in tracks:
            if track['last_frame'] != frame_idx:
                continue  # Only update prediction for tracks present in this frame
            track_id = track['id']
            clamped = clamp_box(track['last_box'], frame.shape)
            if clamped is None:
                continue
            x, y, w, h = clamped
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            # Use 64x64 ROI for RGB mean signal (robust to small misalignment)
            b, g, r = cv2.split(cv2.resize(roi, (64, 64)))
            rgb_mean = [np.mean(r), np.mean(g), np.mean(b)]
            if track_id not in face_rgb_history:
                face_rgb_history[track_id] = []
            face_rgb_history[track_id].append(rgb_mean)
            # For live prediction: use up to the last 150 frames for this track
            rgb_window = np.array(face_rgb_history[track_id][-150:])
            if rgb_window.shape[0] < 10:  # Require a minimum window for reliable rPPG
                continue
            features, hr_bpm, _, _, _ = compute_rppg_features(rgb_window, fs=fps)
            features_scaled = scaler.transform([features])
            prob = clf.predict_proba(features_scaled)[0][1]
            prediction = 'FAKE' if prob > 0.4046 else 'REAL'
            if prediction == 'FAKE':
                fake_count += 1
            else:
                real_count += 1
            all_predictions.append(prob)
            hr_values.append(hr_bpm)
            # Draw predictions and stats on frame
            frame = draw_output(frame, (x, y, w, h), prediction, prob, hr_bpm, frame_idx // int(fps))
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # After processing: plot rPPG for each tracked face
    for track_id, rgb_hist in face_rgb_history.items():
        rgb_arr = np.array(rgb_hist)
        if rgb_arr.shape[0] >= 10:
            features, hr_bpm, rppg_sig, f, pxx = compute_rppg_features(rgb_arr, fs=fps)
            plot_rppg_analysis(rppg_sig, f, pxx, save_dir=os.path.dirname(output_path), track_id=track_id)

    final_prediction = 'FAKE' if fake_count > real_count else 'REAL' if real_count > fake_count else 'UNKNOWN'
    avg_hr = np.mean(hr_values) if hr_values else 0
    avg_conf = np.mean(all_predictions) if all_predictions else 0.0
    return final_prediction, avg_hr, avg_conf, real_count, fake_count
