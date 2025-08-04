import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import json
import collections
import subprocess
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import matplotlib
from filterpy.kalman import KalmanFilter
import sys
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Load the pre-trained deepfake detection model
model = load_model('models/deep_learning_model.keras')

def detect_faces(frame, net, conf=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence > conf:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            if x2 > x1 and y2 > y1:
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

def extract_histogram(frame, box):
    x, y, w, h = box
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return np.zeros(32)
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
class TrackKF:
    def __init__(self, box, frame_idx, frame=None):
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i+4] = 1  # Position + velocity
        self.kf.H = np.zeros((4,8))
        self.kf.H[0,0] = 1
        self.kf.H[1,1] = 1
        self.kf.H[2,2] = 1
        self.kf.H[3,3] = 1
        self.kf.x[:4] = np.array(box).reshape(4, 1)
        self.kf.P *= 10
        self.last_box = box
        self.last_frame = frame_idx
        self.lost_count = 0
        self.appearance = extract_histogram(frame, box) if frame is not None else None
        self.hit_streak = 1
        self.confirmed = False

    def predict(self):
        self.kf.predict()
        pred_box = self.kf.x[:4]
        return tuple(pred_box.astype(int))

    def update(self, box, frame=None):
        self.kf.update(box)
        self.last_box = box
        self.lost_count = 0
        self.hit_streak += 1
        if self.hit_streak >= 3:  # Confirm after 3 consecutive matches
            self.confirmed = True
        if frame is not None:
            self.appearance = extract_histogram(frame, box)

    def mark_missed(self):
        self.lost_count += 1
        self.hit_streak = 0

def robust_track_faces(all_boxes, frames, max_lost=10, iou_threshold=0.3, max_distance=400):
    tracks = {}
    active_tracks = {}  # face_id: TrackKF
    face_id_counter = 0
    for frame_idx, boxes in enumerate(all_boxes):
        # Predict all tracks
        for tid, track in active_tracks.items():
            track.predict()
        # Handle empty frame
        if not boxes:
            for tid in list(active_tracks.keys()):
                active_tracks[tid].mark_missed()
                if active_tracks[tid].lost_count > max_lost:
                    del active_tracks[tid]
            continue
        # Handle first frame or no active tracks
        if not active_tracks:
            for b in boxes:
                tracks[face_id_counter] = [(frame_idx, b)]
                active_tracks[face_id_counter] = TrackKF(b, frame_idx)
                face_id_counter += 1
            continue
        # Get current track info
        track_ids = list(active_tracks.keys())
        track_boxes = np.array([active_tracks[tid].predict() for tid in track_ids])
        n_tracks = len(track_ids)
        n_boxes = len(boxes)
        cost_matrix = np.ones((n_tracks, n_boxes)) * 1000
        box_centroids = np.array([[x + w/2, y + h/2] for (x, y, w, h) in boxes], dtype=np.float32)
        track_centroids = np.array([[b[0] + b[2]/2, b[1] + b[3]/2] for b in track_boxes], dtype=np.float32)

        if box_centroids.shape[0] == 0 or track_centroids.shape[0] == 0:
            # If either is empty, create a (N, 2) empty array
            distances = np.zeros((track_centroids.shape[0], box_centroids.shape[0]))
        else:
            # Always ensure shape (N, 2)
            box_centroids = box_centroids.reshape(-1, 2)
            track_centroids = track_centroids.reshape(-1, 2)
            distances = cdist(track_centroids, box_centroids)
        for i, track_box in enumerate(track_boxes):
            for j, box in enumerate(boxes):
                iou_score = iou(track_box, box)
                distance = distances[i, j]
                # Appearance cost
                track_hist = active_tracks[track_ids[i]].appearance
                box_hist = extract_histogram(frames[frame_idx], box)
                hist_dist = np.linalg.norm(track_hist - box_hist) if track_hist is not None else 0
                # Combine costs
                if iou_score > iou_threshold or distance < max_distance:
                    iou_cost = 1 - iou_score
                    dist_cost = distance / max_distance
                    hist_cost = hist_dist / 10.0  # Normalize histogram distance
                    cost_matrix[i, j] = 0.5 * iou_cost + 0.3 * dist_cost + 0.2 * hist_cost
        if n_tracks > 0 and n_boxes > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_boxes = set()
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.9:
                    tid = track_ids[row]
                    box = boxes[col]
                    tracks.setdefault(tid, []).append((frame_idx, box))
                    active_tracks[tid].update(box)
                    matched_boxes.add(col)
                else:
                    tid = track_ids[row]
                    active_tracks[tid].mark_missed()
            for i, tid in enumerate(track_ids):
                if i not in row_indices or cost_matrix[i, col_indices[list(row_indices).index(i)]] >= 0.9:
                    if active_tracks[tid].last_frame != frame_idx:
                        active_tracks[tid].mark_missed()
            for j, box in enumerate(boxes):
                if j not in matched_boxes:
                    tracks[face_id_counter] = [(frame_idx, box)]
                    active_tracks[face_id_counter] = TrackKF(box, frame_idx)
                    face_id_counter += 1
        for tid in list(active_tracks.keys()):
            track = active_tracks[tid]
            allowed_lost = max_lost if not track.confirmed else max_lost * 2
            if track.lost_count > allowed_lost:
                del active_tracks[tid]
    return tracks

def letterbox_resize(frame, target_size=(640, 360), pad_color=0):
    """
    Resize image to fit within target_size, padding to maintain aspect ratio.
    Returns: padded_image, scaling_factor, padding_offsets
    """
    h, w = frame.shape[:2]
    tw, th = target_size

    # Compute scaling factor (fit to the smallest side)
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Compute padding
    pad_w = tw - new_w
    pad_h = th - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad to target size
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded, scale, (left, top)

def preprocess_frame(frame):
    if frame.size == 0:
        print("Warning: Trying to resize an empty frame.")
        return None
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

def static_video_detection(video_path, output_dir, unique_tag):
    # Check for cache results
    result_path = os.path.join(output_dir, "cached_results.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            cached = json.load(f)
        face_results = cached["face_results"]
        output_path = cached.get("output_path")
        return face_results, output_path

    # Video setup
    output_path = os.path.join(output_dir, f'deep_output.mp4')
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Face detection model
    face_net = cv2.dnn.readNetFromCaffe('models/weights-prototxt.txt', 'models/res_ssd_300Dim.caffeModel')
    def crop_square(frame, x, y, w, h):
        # Center-crop to a square box around the detected face
        size = max(w, h)
        center_x = x + w // 2
        center_y = y + h // 2
        x1 = max(center_x - size // 2, 0)
        y1 = max(center_y - size // 2, 0)
        x2 = min(center_x + size // 2, frame.shape[1])
        y2 = min(center_y + size // 2, frame.shape[0])
        return frame[y1:y2, x1:x2]

    # === Pass 1: Get all face boxes per frame ===
    all_boxes = []
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        padded_frame, scale, (pad_left, pad_top) = letterbox_resize(frame, target_size=(640, 360), pad_color=0)
        boxes = detect_faces(padded_frame, net=face_net)
        mapped_boxes = []
        for x, y, w, h in boxes:
            # Map boxes back to original frame size
            x_orig = int((x - pad_left) / scale)
            y_orig = int((y - pad_top) / scale)
            w_orig = int(w / scale)
            h_orig = int(h / scale)
            mapped_boxes.append((x_orig, y_orig, w_orig, h_orig))
        all_boxes.append(mapped_boxes)
        frames.append(frame.copy())
    cap.release()

    # === Pass 2: Build tracks (track_id: list of (frame_idx, box)) ===
    tracks = robust_track_faces(all_boxes, frames)
    frame_count = len(all_boxes)
    prediction_memory = collections.defaultdict(list)
    
    cap = cv2.VideoCapture(video_path)
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Frame read failed at frame {frame_idx if 'frame_idx' in locals() else '?'}")
            break


        for track_id, detections in tracks.items():
            # Find box in this frame
            this_box = None
            for fidx, box in detections:
                if fidx == frame_idx:
                    this_box = box
                    break
            if this_box is not None:
                x, y, w, h = this_box
                face_img = crop_square(frame, x, y, w, h)
                if face_img.size == 0:
                    continue
                preprocessed_face = preprocess_frame(face_img)
                if preprocessed_face is None:
                    continue
                prediction = model.predict(preprocessed_face)
                score = float(prediction[0][0])
                label = 'Fake' if score > 0.5 else 'Real'
                color = (0, 0, 255) if label == 'Fake' else (0, 255, 0)
                # Store for stats
                prediction_memory[track_id].append((label, score))
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = f"Face {track_id}: {label} ({score:.2f})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                prediction_memory[track_id].append(('Unknown', 0))
        out.write(frame)
    cap.release()
    out.release()

        # === Summary per face, with plots ===
    face_results = []
    for track_id in tracks:
        preds = [p for p, _ in prediction_memory[track_id] if p in ('Real', 'Fake')]
        probs = [prob for _, prob in prediction_memory[track_id]]
        n_real = preds.count('Real')
        n_fake = preds.count('Fake')
        avg_conf = float(np.mean(probs)) if probs else 0.0
        result = "Fake" if n_fake > n_real else "Real"

        # --- Create per-face plots ---
        conf_plot_path, real_v_fake_plot_path = plot_face_track_stats(
            track_id, preds, probs, output_dir, unique_tag
        )

        face_results.append({
            'track_id': track_id,
            'result': result,
            'real_count': n_real,
            'fake_count': n_fake,
            'confidence': round(avg_conf * 100),
            'conf_plot': conf_plot_path,
            'real_vs_fake_plot': real_v_fake_plot_path,
        })

    # Re-encode for compatibility
    current_wd = os.getcwd()
    fixed_output_path = output_path.replace('.mp4', f'_fixed.mp4')
    original_output = os.path.join(current_wd, output_path)
    new_output = os.path.join(current_wd, fixed_output_path)
    subprocess.run([
        'ffmpeg', '-y', '-i', original_output,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart',
        new_output
    ])
    if os.path.exists(original_output):
        os.remove(original_output)
    output_path = fixed_output_path

    result_dict = {
        "face_results": face_results,
        "output_path": output_path
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False)



    return face_results, output_path

# Function to plot and save individual graphs
def plot_face_track_stats(track_id, label_list, score_list, output_folder, unique_tag):
    os.makedirs(output_folder, exist_ok=True)
    # Plot Confidence Score Over Time for this face
    plt.figure(figsize=(8, 3))
    plt.plot(score_list, color='blue')
    plt.title(f'Face {track_id} - Confidence Score Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Confidence Score')
    plt.tight_layout()
    conf_plot_path = os.path.join(output_folder, f'deeplearning_face{track_id}_confidence_{unique_tag}.png')
    plt.savefig(conf_plot_path)
    plt.close()

    # Plot Real vs Fake count for this face
    plt.figure(figsize=(8, 3))
    real_count = label_list.count('Real')
    fake_count = label_list.count('Fake')
    plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    plt.title(f'Face {track_id} - Real vs Fake')
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.tight_layout()
    real_v_fake_plot_path = os.path.join(output_folder, f'deeplearning_face{track_id}_realvfake_{unique_tag}.png')
    plt.savefig(real_v_fake_plot_path)
    plt.close()

    return conf_plot_path, real_v_fake_plot_path

