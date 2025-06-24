import os
import cv2
import numpy as np
import joblib
from scipy.signal import detrend, butter, filtfilt, periodogram
import matplotlib.pyplot as plt
import uuid
import time
import subprocess

clf = joblib.load('models/physio_detection_xgboost_best.pkl')
scaler = joblib.load('models/physio_scaler.pkl')
FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

def detect_faces_dnn(frame, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def track_faces(prev_tracks, curr_boxes, frame_idx, iou_thresh=0.5):
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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return np.zeros_like(signal)
    return filtfilt(b, a, signal)

def plot_rppg_analysis(rppg_sig, f, pxx, real_count, fake_count, save_dir, track_id):
    os.makedirs(save_dir, exist_ok=True)
    plot_id = f"face_{track_id}_{uuid.uuid4().hex[:8]}"
    plt.figure()
    plt.plot(rppg_sig, label='rPPG Signal')
    plt.title(f'Face {track_id} - rPPG Time Series')
    plt.xlabel('Frame Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    rppg_signal_plot_path = os.path.join(save_dir, f'rppg_signal_{plot_id}.png')
    plt.savefig(rppg_signal_plot_path)
    plt.close()
    plt.figure()
    plt.plot(f, pxx, label='Power Spectrum')
    plt.title(f'Face {track_id} - rPPG Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.tight_layout()
    rppg_spectrum_plot_path = os.path.join(save_dir, f'rppg_spectrum_{plot_id}.png')
    plt.savefig(rppg_spectrum_plot_path)
    plt.close()
    plt.figure(figsize=(8, 3))
    plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    plt.title('Counts of Real and Fake Predictions')
    plt.xlabel('Prediction Type')
    plt.ylabel('Count')
    plt.tight_layout()
    prediction_count_path = os.path.join(save_dir, f'prediction_count_{plot_id}.png')
    plt.savefig(prediction_count_path)
    plt.clf()
    return rppg_signal_plot_path, rppg_spectrum_plot_path, prediction_count_path

def compute_rppg_features(rgb_signal, fs):
    rgb_detrend = detrend(rgb_signal, axis=0)
    rgb_norm = (rgb_detrend - np.mean(rgb_detrend, axis=0)) / (np.std(rgb_detrend, axis=0) + 1e-8)
    S = rgb_norm
    h = (S[:, 1] - S[:, 0]) / (np.std(S[:, 1] - S[:, 0]) + 1e-8)
    s = (S[:, 1] + S[:, 0] - 2 * S[:, 2]) / (np.std(S[:, 1] + S[:, 0] - 2 * S[:, 2]) + 1e-8)
    rppg_sig = h + s
    if len(rppg_sig) <= 21:
        return [0]*9, 0, np.zeros_like(rppg_sig), np.zeros(1), np.zeros(1)
    rppg_sig = butter_bandpass_filter(rppg_sig, fs)
    f, pxx = periodogram(rppg_sig, fs)
    valid = (f >= 0.7) & (f <= 4.0)
    f, pxx = f[valid], pxx[valid]
    if len(f) == 0:
        return [0]*9, 0, np.zeros_like(rppg_sig), np.zeros(1), np.zeros(1)
    peak_idx = np.argmax(pxx)
    hr_freq, hr_bpm, hr_power = f[peak_idx], f[peak_idx]*60, pxx[peak_idx]
    snr = hr_power / (np.sum(pxx) - hr_power + 1e-8)
    features = [np.mean(rppg_sig), np.std(rppg_sig), np.max(rppg_sig), np.min(rppg_sig),
                np.ptp(rppg_sig), hr_freq, hr_bpm, hr_power, snr]
    return features, hr_bpm, rppg_sig, f, pxx

def draw_output(frame, face_box, prediction, confidence, hr_bpm, time_stamp, track_id):
    x, y, w, h = face_box
    color = (0, 255, 0) if prediction == 'REAL' else (0, 0, 255)
    label = f"Face {track_id}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y_text = y + h + 25
    cv2.putText(frame, f"{prediction.upper()} - HR: {hr_bpm:.2f} BPM", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_text += 25
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_text += 25
    cv2.putText(frame, f"Time: {time_stamp}s", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def clamp_box(box, frame_shape):
    x, y, w, h = box
    H, W = frame_shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)

def get_video_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30
    return fps

def run_detection(source, output_path=f'static/results/output.mp4', is_webcam=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(0 if is_webcam else source)
    fps = get_video_fps(cap)
    width, height = int(cap.get(3)), int(cap.get(4))

    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        test_writer = cv2.VideoWriter('test_h264.mp4', fourcc, fps, (width, height))
        if not test_writer.isOpened():
            raise RuntimeError('H.264 codec not available.')
        test_writer.release()
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    tracks = []
    face_rgb_history = {}
    face_pred_history = {}
    face_prob_history = {}
    face_hr_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_faces_dnn(frame)
        tracks = track_faces(tracks, boxes, frame_idx)
        for track in tracks:
            if track['last_frame'] != frame_idx:
                continue
            track_id = track['id']
            clamped = clamp_box(track['last_box'], frame.shape)
            if clamped is None:
                continue
            x, y, w, h = clamped
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            b, g, r = cv2.split(cv2.resize(roi, (64, 64)))
            rgb_mean = [np.mean(r), np.mean(g), np.mean(b)]
            if track_id not in face_rgb_history:
                face_rgb_history[track_id] = []
                face_pred_history[track_id] = []
                face_prob_history[track_id] = []
                face_hr_history[track_id] = []
            face_rgb_history[track_id].append(rgb_mean)
            rgb_window = np.array(face_rgb_history[track_id][-150:])
            if rgb_window.shape[0] < 10:
                continue
            features, hr_bpm, _, _, _ = compute_rppg_features(rgb_window, fs=fps)
            features_scaled = scaler.transform([features])
            prob = clf.predict_proba(features_scaled)[0][1]
            prediction = 'FAKE' if prob > 0.4046 else 'REAL'
            face_pred_history[track_id].append(prediction)
            face_prob_history[track_id].append(prob)
            face_hr_history[track_id].append(hr_bpm)
            frame = draw_output(frame, (x, y, w, h), prediction, prob, hr_bpm, frame_idx // int(fps), track_id)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    time.sleep(0.2)  # Short pause to ensure file is closed

    # === ffmpeg PATCH for HTML5 playback compatibility ===
    fixed_output_path = output_path.replace('.mp4', '_fixed.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', output_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart',
        fixed_output_path
    ])
    if os.path.exists(output_path):
        os.remove(output_path)
    output_path = fixed_output_path

    face_results = []
    MIN_PREDICTIONS = 30
    for track_id, preds in face_pred_history.items():
        if len(preds) < MIN_PREDICTIONS:
            continue
        n_real = preds.count('REAL')
        n_fake = preds.count('FAKE')
        if n_real > n_fake:
            result = 'Real'
        elif n_fake > n_real:
            result = 'Fake'
        else:
            result = 'Unknown'
        avg_hr = np.mean(face_hr_history[track_id]) if face_hr_history[track_id] else 0
        avg_conf = np.mean(face_prob_history[track_id]) if face_prob_history[track_id] else 0.0
        avg_conf = round(avg_conf * 100)
        rgb_arr = np.array(face_rgb_history[track_id])
        if rgb_arr.shape[0] >= MIN_PREDICTIONS:
            features, hr_bpm, rppg_sig, f, pxx = compute_rppg_features(rgb_arr, fs=fps)
            signal_plot, spectrum_plot, pred_count_plot = plot_rppg_analysis(
                rppg_sig, f, pxx, n_real, n_fake, save_dir=os.path.dirname(output_path), track_id=track_id)
        else:
            signal_plot, spectrum_plot, pred_count_plot = None, None, None
        face_results.append({
            'track_id': track_id,
            'result': result,
            'real_count': n_real,
            'fake_count': n_fake,
            'hr': round(avg_hr),
            'confidence': avg_conf,
            'signal_plot': signal_plot,
            'spectrum_plot': spectrum_plot,
            'pred_count_plot': pred_count_plot
        })

    return face_results, output_path
