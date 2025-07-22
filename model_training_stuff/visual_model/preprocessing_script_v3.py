import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random

# Configuration
FAKE_VID_DIR = "/mnt/d/fake_videos"
REAL_VID_DIR = "/mnt/d/real_videos"
OUTPUT_DIR = "/mnt/d/preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_VIDEOS = None
FRAME_INTERVAL = 0.5
FACE_SIZE = 64

AUGMENT_PROB = 0.5  # Increase to 50% for more augmentation
BLUR_KERNEL = (3, 3)
NOISE_STDDEV = 7.0  # Slightly more noise

# Load OpenCV DNN face detector (ResNet10-SSD model in Caffe format)
PROTO_PATH = "models/weights-prototxt.txt"  # path to deploy.prototxt file
MODEL_PATH = "models/res_ssd_300Dim.caffeModel"    # path to caffemodel
if not os.path.isfile(PROTO_PATH) or not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("Face detector model files not found. Please download deploy.prototxt and .caffemodel.")

face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def apply_brightness_contrast(img, brightness=0, contrast=0):
    # Randomly change brightness and contrast
    beta = np.random.randint(-30, 30) if brightness == 0 else brightness
    alpha = 1 + 0.2 * (np.random.rand() - 0.5) if contrast == 0 else contrast
    img = img.astype(np.float32) * alpha + beta
    return np.clip(img, 0, 255).astype('uint8')

def apply_cutout(img, size=12):
    # Randomly mask a rectangle region
    h, w = img.shape
    top = np.random.randint(0, h - size)
    left = np.random.randint(0, w - size)
    img[top:top+size, left:left+size] = 0
    return img

def apply_augmentations(img):
    # Apply a random combination of augmentations
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip
    if np.random.rand() < 0.5:
        img = apply_brightness_contrast(img)
    if np.random.rand() < 0.4:
        img = cv2.GaussianBlur(img, BLUR_KERNEL, 0)
    if np.random.rand() < 0.4:
        img = apply_cutout(img)
    if np.random.rand() < 0.4:
        noise = np.random.normal(0, NOISE_STDDEV, img.shape).astype('float32')
        img = np.clip(img.astype('float32') + noise, 0, 255).astype('uint8')
    return img

def extract_hog_features(img_gray):
    return hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

def extract_lbp_features(img_gray):
    lbp_r1 = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    lbp_r2 = local_binary_pattern(img_gray, P=16, R=2, method='uniform')
    hist_r1, _ = np.histogram(lbp_r1, bins=np.arange(0, 8+2), range=(0, 8+1))
    hist_r2, _ = np.histogram(lbp_r2, bins=np.arange(0, 16+2), range=(0, 16+1))
    hist_r1 = hist_r1.astype('float32')
    hist_r2 = hist_r2.astype('float32')
    if hist_r1.sum() != 0:
        hist_r1 /= hist_r1.sum()
    if hist_r2.sum() != 0:
        hist_r2 /= hist_r2.sum()
    return np.concatenate([hist_r1, hist_r2])

def process_video(video_path, label, batch_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: unable to open video {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_frames = int(round(fps * FRAME_INTERVAL))
    features_list = []
    frame_count = 0
    success = True
    while success:
        success = cap.grab()
        if not success:
            break
        if frame_count % frame_interval_frames == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                frame_count += 1
                continue
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         (104.0, 117.0, 123.0), swapRB=False, crop=False)
            face_net.setInput(blob)
            detections = face_net.forward()
            best_conf = 0
            best_box = None
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.5 and conf > best_conf:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype('int')
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 - x1 > 0 and y2 - y1 > 0:
                        best_conf = conf
                        best_box = (x1, y1, x2, y2)
            if best_box is None:
                frame_count += 1
                continue
            (x1, y1, x2, y2) = best_box
            face = frame[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            # Balanced augmentation for both real and fake:
            if np.random.rand() < AUGMENT_PROB:
                face_gray = apply_augmentations(face_gray)
            hog_vec = extract_hog_features(face_gray)
            lbp_vec = extract_lbp_features(face_gray)
            feat_vec = np.concatenate([hog_vec, lbp_vec]).astype('float32')
            if np.var(feat_vec) < 1e-6:
                frame_count += 1
                continue
            features_list.append(feat_vec)
        frame_count += 1
    cap.release()
    if not features_list:
        return None
    features_arr = np.stack(features_list, axis=0)
    labels_arr = np.full((features_arr.shape[0],), label, dtype='uint8')
    cls = 'fake' if label == 1 else 'real'
    out_path = os.path.join(OUTPUT_DIR, f"{cls}_batch_{batch_index:04d}.npz")
    np.savez_compressed(out_path, features=features_arr, labels=labels_arr)
    return out_path

def process_video_wrapper(args):
    return process_video(*args)

# --- Main execution ---
if __name__ == "__main__":
    fake_videos = sorted([os.path.join(FAKE_VID_DIR, f) for f in os.listdir(FAKE_VID_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
    real_videos = sorted([os.path.join(REAL_VID_DIR, f) for f in os.listdir(REAL_VID_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
    total_fakes = len(fake_videos)
    total_reals = len(real_videos)
    if MAX_VIDEOS is not None:
        per_class = MAX_VIDEOS // 2
        total_fakes = min(total_fakes, per_class)
        total_reals = min(total_reals, per_class)
    else:
        min_count = min(total_fakes, total_reals)
        total_fakes = total_reals = min_count
    fake_videos = fake_videos[:total_fakes]
    real_videos = real_videos[:total_reals]
    print(f"Processing {len(fake_videos)} fake videos and {len(real_videos)} real videos...")

    tasks = []
    batch_index = 0
    for vid in fake_videos:
        out_path = os.path.join(OUTPUT_DIR, f"fake_batch_{batch_index:04d}.npz")
        if os.path.exists(out_path):
            batch_index += 1
            continue
        tasks.append((vid, 1, batch_index))
        batch_index += 1
    for vid in real_videos:
        out_path = os.path.join(OUTPUT_DIR, f"real_batch_{batch_index:04d}.npz")
        if os.path.exists(out_path):
            batch_index += 1
            continue
        tasks.append((vid, 0, batch_index))
        batch_index += 1

    # ---- tqdm progress bar for Pool mapping ----
    num_workers = 8
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_video_wrapper, tasks),
                      total=len(tasks), desc="Preprocessing Videos"):
            pass

    print("Preprocessing completed. Outputs saved to", OUTPUT_DIR)
