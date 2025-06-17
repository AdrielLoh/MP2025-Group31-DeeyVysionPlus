import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

SOURCE_DIR = "G:/deepfake_training_datasets/DeeperForensics/legacy_files/potential_multiple_faces_FAKE-ONLY"
DEST_DIR = "G:/deepfake_training_datasets/DeeperForensics/training/fake"
os.makedirs(DEST_DIR, exist_ok=True)

FACE_PROTO = 'models/weights-prototxt.txt'
FACE_MODEL = 'models/res_ssd_300Dim.caffeModel'

def detect_faces_dnn(frame, conf_threshold=0.5):
    net = get_face_net()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, boxA[2] * boxA[3])
    boxBArea = max(1, boxB[2] * boxB[3])
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou_val

def track_faces(all_face_boxes, iou_thresh=0.5):
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
    tracklets = {t['id']: t['boxes'] for t in tracks if len(t['boxes']) > 2}
    return tracklets

def get_face_net():
    net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    return net

def process_video(video_path):
    net = get_face_net()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (video_path, None)

    all_faces = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_faces_dnn(frame, conf_threshold=0.5)
        all_faces.append(boxes)
    cap.release()

    face_tracks = track_faces(all_faces)
    return (video_path, len(face_tracks))

def main():
    video_files = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} video files. Processing...")

    max_cpu = max(1, multiprocessing.cpu_count() - 6)

    with ProcessPoolExecutor(max_workers=max_cpu) as executor:
        future_to_video = {executor.submit(process_video, vid): vid for vid in video_files}

        for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Processing"):
            video_path, face_count = future.result()

            if face_count == 1:
                filename = os.path.basename(video_path)
                dest_path = os.path.join(DEST_DIR, filename)
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    count = 1
                    while True:
                        new_name = f"{base}_{count}{ext}"
                        dest_path = os.path.join(DEST_DIR, new_name)
                        if not os.path.exists(dest_path):
                            break
                        count += 1
                shutil.move(video_path, dest_path)
                print(f"Moved: {video_path} -> {dest_path}")

if __name__ == "__main__":
    main()
