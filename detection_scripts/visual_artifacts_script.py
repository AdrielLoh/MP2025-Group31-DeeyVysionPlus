import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import os
import uuid
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO)

# Setup
FACE_LANDMARKER_PATH = 'models/face_landmarker.task'
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode
landmarker_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)
net = cv2.dnn.readNetFromCaffe('models/weights-prototxt.txt', 'models/res_ssd_300Dim.caffeModel')
model = tf.keras.models.load_model('models/visual_artifacts.keras')

def mediapipe_detections(image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = face_landmarker.detect(mp_image)
    boxes = []
    all_landmarks = []
    if results and results.face_landmarks:
        h, w, _ = image.shape
        for landmarks in results.face_landmarks:
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            box = [x_min, y_min, x_max - x_min, y_max - y_min]
            boxes.append(box)
            all_landmarks.append(landmarks)
    return boxes, all_landmarks

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, boxA[2] * boxA[3])
    boxBArea = max(1, boxB[2] * boxB[3])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def track_faces_multi(all_face_boxes, iou_thresh=0.5):
    tracks = []
    next_track_id = 0
    for frame_idx, boxes in enumerate(all_face_boxes):
        assigned = [False] * len(boxes)
        for track in tracks:
            found = False
            for i, box in enumerate(boxes):
                if not assigned[i] and iou(track['last_box'], box) > iou_thresh and frame_idx - track['last_frame'] == 1:
                    track['boxes'].append((frame_idx, box))
                    track['last_box'] = box
                    track['last_frame'] = frame_idx
                    assigned[i] = True
                    found = True
                    break
        for i, box in enumerate(boxes):
            if not assigned[i]:
                tracks.append({'boxes': [(frame_idx, box)], 'last_box': box, 'last_frame': frame_idx, 'id': next_track_id})
                next_track_id += 1
    # Return {track_id: [(frame_idx, box), ...]}
    return {t['id']: t['boxes'] for t in tracks if len(t['boxes']) > 2}

def merge_tracks_if_single_face(tracklets, total_frames):
    """
    If only one face appears per frame, but due to tracking breaks multiple tracks exist,
    merge them all into one.
    """
    # Map: frame_idx -> (track_id, box)
    per_frame = {}
    for tid, track in tracklets.items():
        for (fidx, box) in track:
            per_frame[fidx] = (tid, box)
    # Are there frames with more than one face?
    max_faces = max([sum(1 for tid, track in tracklets.items() if any(fidx == f for f, _ in track)) for fidx in range(total_frames)])
    if max_faces > 1 or len(tracklets) == 1:
        return tracklets  # keep all separate
    # Otherwise, combine all tracks into one "supertrack"
    frames_and_boxes = []
    for tid, track in tracklets.items():
        frames_and_boxes.extend(track)
    frames_and_boxes.sort()  # by frame index
    return {0: frames_and_boxes}

def extract_features(image, net, box, landmarks, fixed_length=4096):
    try:
        h, w, _ = image.shape
        landmark_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        landmark_coords = normalize(landmark_coords.reshape(1, -1)).flatten()
        x, y, bw, bh = box
        x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        face_crop = image[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else image
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        dnn_features = net.forward().flatten()
        combined_features = np.hstack((landmark_coords, dnn_features))
        if len(combined_features) > fixed_length:
            combined_features = combined_features[:fixed_length]
        elif len(combined_features) < fixed_length:
            combined_features = np.pad(combined_features, (0, fixed_length - len(combined_features)), 'constant')
        # Debug: print feature stats
        logging.info(f'Feature stats: min={np.min(combined_features):.4f}, max={np.max(combined_features):.4f}, mean={np.mean(combined_features):.4f}, shape={combined_features.shape}')
        if np.any(np.isnan(combined_features)):
            logging.warning('Features contain NaNs!')
        if np.all(combined_features == 0):
            logging.warning('Features are all zeros!')
        return combined_features, np.linalg.norm(landmark_coords), np.linalg.norm(dnn_features)
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None, None, None

def plot_face_metrics(metrics, output_dir, track_id, uid):
    os.makedirs(output_dir, exist_ok=True)
    base = f"face_{track_id}_{uid}"
    chart_paths = []
    # 1. Prediction Timeline
    plt.figure(figsize=(10, 4))
    plt.plot(metrics['frame_indices'], metrics['labels'], drawstyle='steps-post')
    plt.title('Prediction Timeline')
    plt.xlabel('Frame')
    plt.ylabel('Label (1=Real, 0=Fake)')
    plt.tight_layout()
    timeline_path = os.path.join(output_dir, f'timeline_{base}.png')
    plt.savefig(timeline_path, dpi=120)
    plt.close()
    chart_paths.append(timeline_path)
    # 2. Confidence Over Time
    plt.figure(figsize=(10, 4))
    plt.plot(metrics['frame_indices'], metrics['confidences'])
    plt.title('Confidence Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Model Confidence')
    plt.tight_layout()
    confidence_path = os.path.join(output_dir, f'confidence_{base}.png')
    plt.savefig(confidence_path, dpi=120)
    plt.close()
    chart_paths.append(confidence_path)
    # 3. DNN Feature Norms
    plt.figure(figsize=(10, 4))
    plt.plot(metrics['frame_indices'], metrics['dnn_norms'])
    plt.title('DNN Feature Norms')
    plt.xlabel('Frame')
    plt.ylabel('Norm')
    plt.tight_layout()
    dnn_path = os.path.join(output_dir, f'dnn_{base}.png')
    plt.savefig(dnn_path, dpi=120)
    plt.close()
    chart_paths.append(dnn_path)
    # 4. Detection Time Per Frame
    plt.figure(figsize=(10, 4))
    plt.plot(metrics['frame_indices'], metrics['detection_times'])
    plt.title('Detection Time per Frame')
    plt.xlabel('Frame')
    plt.ylabel('Seconds')
    plt.tight_layout()
    time_path = os.path.join(output_dir, f'detecttime_{base}.png')
    plt.savefig(time_path, dpi=120)
    plt.close()
    chart_paths.append(time_path)
    # 5. Real vs Fake Frame Count Bar Chart
    plt.figure(figsize=(5, 4))
    real_count = metrics['labels'].count(1)
    fake_count = metrics['labels'].count(0)
    plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    plt.title('Frame Count: Real vs Fake')
    plt.ylabel('Frame Count')
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f'framecount_{base}.png')
    plt.savefig(bar_path, dpi=120)
    plt.close()
    chart_paths.append(bar_path)
    return chart_paths

def run_visual_artifacts_detection(video_path, video_tag, output_dir='static/results', min_frames=5, method="single"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    unique_tag = video_tag
    output_video_path = os.path.join(output_dir, f'visual_artifacts_overlay_{unique_tag}.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    all_face_boxes = []
    frame_landmarks = []
    frame_buffer = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes, all_landmarks = mediapipe_detections(frame)
        all_face_boxes.append(boxes)
        frame_landmarks.append(all_landmarks)
        frame_buffer.append(frame.copy())
        frame_idx += 1
    cap.release()
    total_frames = len(frame_buffer)
    # Tracking
    tracklets = track_faces_multi(all_face_boxes)
    tracklets = merge_tracks_if_single_face(tracklets, total_frames)
    face_results = []
    # For each track/face
    for track_id, track in tracklets.items():
        metrics = {
            'frame_indices': [],
            'confidences': [],
            'labels': [],
            'dnn_norms': [],
            'detection_times': []
        }
        prev_features = None
        for (frame_idx, box) in track:
            frame = frame_buffer[frame_idx]
            all_lm = frame_landmarks[frame_idx]
            match_lm = None
            for i, b in enumerate(all_face_boxes[frame_idx]):
                if np.linalg.norm(np.array(b) - np.array(box)) < 5:
                    match_lm = all_lm[i]
                    break
            if match_lm is None:
                logging.warning(f"No matching landmarks for frame {frame_idx}, box {box}. Skipping frame.")
                continue
            start_time = time.time()
            features, landmark_norm, dnn_norm = extract_features(frame, net, box, match_lm)
            detect_time = time.time() - start_time
            if features is not None:
                if prev_features is not None and np.array_equal(features, prev_features):
                    logging.warning(f"Features for frame {frame_idx} are IDENTICAL to previous frame.")
                prev_features = features.copy()
                # Model prediction
                model_input = np.expand_dims(features, axis=0)
                pred = float(model.predict(model_input, verbose=0).squeeze())
                logging.info(f'Frame {frame_idx} model output: {pred:.4f}')
                label = 1 if pred >= 0.8 else 0
                metrics['frame_indices'].append(frame_idx)
                metrics['confidences'].append(pred)
                metrics['labels'].append(label)
                metrics['dnn_norms'].append(dnn_norm)
                metrics['detection_times'].append(detect_time)
            else:
                logging.warning(f"Feature extraction failed for frame {frame_idx}.")
        if len(metrics['frame_indices']) < min_frames:
            logging.warning(f"Not enough frames for track {track_id}, skipping.")
            continue
        # Overlay drawing (only bounding box, no mesh, no face landmarks)
        for idx, frame_idx in enumerate(metrics['frame_indices']):
            frame = frame_buffer[frame_idx]
            box = track[idx][1]
            color = (0, 255, 0) if metrics['labels'][idx] == 1 else (0, 0, 255)
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_txt = f"{'REAL' if metrics['labels'][idx] == 1 else 'FAKE'}: {metrics['confidences'][idx]:.2f}"
            cv2.putText(frame, label_txt, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            frame_buffer[frame_idx] = frame
        chart_paths = plot_face_metrics(metrics, output_dir, track_id, video_tag)
        real_count = metrics['labels'].count(1)
        fake_count = metrics['labels'].count(0)
        avg_conf = round(100 * (sum(metrics['confidences']) / len(metrics['confidences'])), 1)
        face_results.append({
            'track_id': track_id,
            'real_count': real_count,
            'fake_count': fake_count,
            'confidence': avg_conf,
            'charts': [os.path.relpath(p, 'static').replace('\\', '/') for p in chart_paths],
            'result': 'Real' if real_count > fake_count else 'Fake',
        })
    # Save overlay video
    for frame in frame_buffer:
        out.write(frame)
    out.release()
    # ffmpeg fix for browser
    fixed_path = output_video_path.replace('.mp4', f'_fixed_{unique_tag}.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', output_video_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart',
        fixed_path
    ])

    if os.path.exists(output_video_path):
        os.remove(output_video_path)
    output_video_path = fixed_path

    # Clean up uploads folder
    if method != "multi":
        if os.path.exists(video_path):
            os.remove(video_path)
    
    return face_results, output_video_path
