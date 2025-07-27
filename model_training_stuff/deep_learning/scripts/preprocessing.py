import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

# ====== CONFIGURATION ======
DATA_DIRS = [
    r'C:\deepvysion\manipulated_videos',
    r'C:\deepvysion\source_videos'
]
OUTPUT_DIR = r'C:\deepvysion\preprocessed_batches_pri'
BATCH_SIZE = 40
FRAME_SIZE = (128, 128)
FRAMES_PER_VIDEO = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return frame[y:y+h, x:x+w]

def get_label_from_path(path):
    lower = path.lower()
    if any(x in lower for x in ['manipulated', 'deepfake', 'face2face', 'faceshifter', 'faceswap', 'neuraltextures']):
        return 1
    if 'source' in lower or 'real' in lower:
        return 0
    return None

def collect_all_videos():
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    labels = []
    for data_dir in DATA_DIRS:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    fpath = os.path.join(root, file)
                    video_files.append(fpath)
                    labels.append(get_label_from_path(fpath))
    return list(zip(video_files, labels))

def preprocess_video(args):
    video_path, label = args
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return (video_path, None, label)
        sample_idxs = np.linspace(0, total_frames-1, FRAMES_PER_VIDEO, dtype=int)
        idx_set = set(sample_idxs)
        idx = 0
        success, frame = cap.read()
        while success:
            if idx in idx_set:
                face = detect_and_crop_face(frame)
                if face is not None:
                    face = cv2.resize(face, FRAME_SIZE)
                    face = face.astype(np.float32) / 255.0
                    frames.append(face)
            success, frame = cap.read()
            idx += 1
        cap.release()
        if len(frames) == 0:
            return (video_path, None, label)
        return (video_path, np.array(frames), label)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return (video_path, None, label)

def save_batch(batch_data, batch_index):
    batch_frames, batch_labels, batch_paths = [], [], []
    for video_path, frames, label in batch_data:
        if frames is not None and len(frames) > 0:
            batch_frames.append(frames)
            batch_labels.append(label)
            batch_paths.append(video_path)
    if batch_frames:
        np.save(os.path.join(OUTPUT_DIR, f'batch_{batch_index}_frames.npy'), np.array(batch_frames, dtype=object))
        np.save(os.path.join(OUTPUT_DIR, f'batch_{batch_index}_labels.npy'), np.array(batch_labels))
        with open(os.path.join(OUTPUT_DIR, f'batch_{batch_index}_videos.txt'), 'w') as f:
            for vp in batch_paths:
                f.write(vp + '\n')

def main():
    all_videos = collect_all_videos()
    print(f'Found {len(all_videos)} videos to process.')
    batch_index = 698
    for i in range(0, len(all_videos), BATCH_SIZE):
        batch_videos = all_videos[i:i+BATCH_SIZE]
        print(f'Processing batch {batch_index} with {len(batch_videos)} videos...')
        with Pool(cpu_count()) as pool:
            batch_data = list(tqdm(pool.imap(preprocess_video, batch_videos), total=len(batch_videos)))
        save_batch(batch_data, batch_index)
        print(f'Batch {batch_index} saved.')
        batch_index += 1

if __name__ == '__main__':
    main()
