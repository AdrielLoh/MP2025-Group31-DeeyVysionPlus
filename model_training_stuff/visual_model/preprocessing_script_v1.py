# preprocess.py – Preprocess videos to extract face images and features
import os
import cv2
import uuid
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.preprocessing import normalize
import concurrent.futures
import argparse
import logging
import random

logging.basicConfig(level=logging.INFO)

# Directories and model paths
REAL_VID_DIR = "/mnt/d/real_videos"
FAKE_VID_DIR = "/mnt/d/fake_videos"
OUTPUT_DIR = "/mnt/d/preprocessed_data"
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
REAL_IMG_DIR = os.path.join(IMAGE_OUTPUT_DIR, "real")
FAKE_IMG_DIR = os.path.join(IMAGE_OUTPUT_DIR, "fake")
# Paths to face detector model (Caffe SSD)
CAFFE_PROTO_PATH = "models/weights-protxt.txt"  # path for caffe model weight
CAFFE_MODEL_PATH = "models/res_ssd_300Dim.caffeModel"  # path to caffe model 
# Path to MediaPipe face landmarker model
FACE_LANDMARKER_PATH = "models/face_landmarker.task"

# Globals for model instances (to be initialized in worker processes)
face_net = None
face_landmarker = None

def process_video(video_path, label, frame_step=1, do_augment=True):
    """Process a single video: detect faces, extract features (and augment)."""
    global face_net, face_landmarker
    # Initialize models once per process
    if face_net is None:
        face_net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO_PATH, CAFFE_MODEL_PATH)
    if face_landmarker is None:
        BaseOptions = python.BaseOptions
        VisionRunningMode = vision.RunningMode
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    features_list = []
    labels_list = []
    frame_idx = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_step > 1 and frame_idx % frame_step != 0:
            frame_idx += 1
            continue
        # Run MediaPipe face detection & landmarks
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = face_landmarker.detect(mp_frame)
        if results and results.face_landmarks:
            h, w, _ = frame.shape
            for lm_list in results.face_landmarks:
                # Compute face bounding box from landmarks
                xs = [lm.x * w for lm in lm_list]
                ys = [lm.y * h for lm in lm_list]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                x1, y1 = max(0, x_min), max(0, y_min)
                x2, y2 = min(w, x_max), min(h, y_max)
                face_crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
                # Landmark coordinate vector (normalized)
                lm_coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list]).flatten()
                lm_coords = normalize(lm_coords.reshape(1, -1)).flatten()
                # CNN features from face detector
                blob = cv2.dnn.blobFromImage(face_crop, 1.0, (300, 300),
                                             (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                dnn_out = face_net.forward().flatten()
                # Combine features
                combined = np.hstack((lm_coords, dnn_out))
                if combined.shape[0] < 4096:
                    combined = np.pad(combined, (0, 4096 - combined.shape[0]), 'constant')
                else:
                    combined = combined[:4096]
                features_list.append(combined)
                labels_list.append(1 if label == "real" else 0)
                # Save face image
                img_dir = REAL_IMG_DIR if label == "real" else FAKE_IMG_DIR
                os.makedirs(img_dir, exist_ok=True)
                img_filename = f"{label}_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(os.path.join(img_dir, img_filename), face_crop)
                if do_augment:
                    # --- 1. Horizontal Flip ---
                    flip_img = cv2.flip(face_crop, 1)
                    mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB))
                    flip_results = face_landmarker.detect(mp_flip)
                    if flip_results and flip_results.face_landmarks:
                        lm_list_flip = flip_results.face_landmarks[0]
                        lm_coords_flip = np.array([[lm.x, lm.y, lm.z] for lm in lm_list_flip]).flatten()
                        lm_coords_flip = normalize(lm_coords_flip.reshape(1, -1)).flatten()
                        blob_flip = cv2.dnn.blobFromImage(flip_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                        face_net.setInput(blob_flip)
                        dnn_out_flip = face_net.forward().flatten()
                        combined_flip = np.hstack((lm_coords_flip, dnn_out_flip))
                        if combined_flip.shape[0] < 4096:
                            combined_flip = np.pad(combined_flip, (0, 4096 - combined_flip.shape[0]), 'constant')
                        else:
                            combined_flip = combined_flip[:4096]
                        features_list.append(combined_flip)
                        labels_list.append(1 if label == "real" else 0)
                        img_filename_flip = f"{label}_{uuid.uuid4().hex}_flip.jpg"
                        cv2.imwrite(os.path.join(img_dir, img_filename_flip), flip_img)

                    # --- 2. Brightness Adjustment ---
                    factor = np.random.uniform(0.5, 1.5)
                    bright_img = cv2.convertScaleAbs(face_crop, alpha=factor, beta=0)
                    mp_bright = mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB))
                    bright_results = face_landmarker.detect(mp_bright)
                    if bright_results and bright_results.face_landmarks:
                        lm_list_bright = bright_results.face_landmarks[0]
                        lm_coords_bright = np.array([[lm.x, lm.y, lm.z] for lm in lm_list_bright]).flatten()
                        lm_coords_bright = normalize(lm_coords_bright.reshape(1, -1)).flatten()
                        blob_bright = cv2.dnn.blobFromImage(bright_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                        face_net.setInput(blob_bright)
                        dnn_out_bright = face_net.forward().flatten()
                        combined_bright = np.hstack((lm_coords_bright, dnn_out_bright))
                        if combined_bright.shape[0] < 4096:
                            combined_bright = np.pad(combined_bright, (0, 4096 - combined_bright.shape[0]), 'constant')
                        else:
                            combined_bright = combined_bright[:4096]
                        features_list.append(combined_bright)
                        labels_list.append(1 if label == "real" else 0)
                        img_filename_br = f"{label}_{uuid.uuid4().hex}_bright.jpg"
                        cv2.imwrite(os.path.join(img_dir, img_filename_br), bright_img)

                    # --- 3. Rotation ---
                    angle = np.random.uniform(-10, 10)
                    rot_mat = cv2.getRotationMatrix2D(((x2-x1)//2, (y2-y1)//2), angle, 1.0)
                    rotated_img = cv2.warpAffine(face_crop, rot_mat, (x2-x1, y2-y1), borderMode=cv2.BORDER_REFLECT)
                    mp_rot = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
                    rot_results = face_landmarker.detect(mp_rot)
                    if rot_results and rot_results.face_landmarks:
                        lm_list_rot = rot_results.face_landmarks[0]
                        lm_coords_rot = np.array([[lm.x, lm.y, lm.z] for lm in lm_list_rot]).flatten()
                        lm_coords_rot = normalize(lm_coords_rot.reshape(1, -1)).flatten()
                        blob_rot = cv2.dnn.blobFromImage(rotated_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                        face_net.setInput(blob_rot)
                        dnn_out_rot = face_net.forward().flatten()
                        combined_rot = np.hstack((lm_coords_rot, dnn_out_rot))
                        if combined_rot.shape[0] < 4096:
                            combined_rot = np.pad(combined_rot, (0, 4096 - combined_rot.shape[0]), 'constant')
                        else:
                            combined_rot = combined_rot[:4096]
                        features_list.append(combined_rot)
                        labels_list.append(1 if label == "real" else 0)
                        img_filename_rot = f"{label}_{uuid.uuid4().hex}_rot.jpg"
                        cv2.imwrite(os.path.join(img_dir, img_filename_rot), rotated_img)

                    # --- 4. Contrast Adjustment ---
                    contrast_factor = np.random.uniform(0.7, 1.3)
                    contrast_img = cv2.convertScaleAbs(face_crop, alpha=contrast_factor, beta=0)
                    mp_contrast = mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB))
                    contrast_results = face_landmarker.detect(mp_contrast)
                    if contrast_results and contrast_results.face_landmarks:
                        lm_list_contrast = contrast_results.face_landmarks[0]
                        lm_coords_contrast = np.array([[lm.x, lm.y, lm.z] for lm in lm_list_contrast]).flatten()
                        lm_coords_contrast = normalize(lm_coords_contrast.reshape(1, -1)).flatten()
                        blob_contrast = cv2.dnn.blobFromImage(contrast_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                        face_net.setInput(blob_contrast)
                        dnn_out_contrast = face_net.forward().flatten()
                        combined_contrast = np.hstack((lm_coords_contrast, dnn_out_contrast))
                        if combined_contrast.shape[0] < 4096:
                            combined_contrast = np.pad(combined_contrast, (0, 4096 - combined_contrast.shape[0]), 'constant')
                        else:
                            combined_contrast = combined_contrast[:4096]
                        features_list.append(combined_contrast)
                        labels_list.append(1 if label == "real" else 0)
                        img_filename_contrast = f"{label}_{uuid.uuid4().hex}_contrast.jpg"
                        cv2.imwrite(os.path.join(img_dir, img_filename_contrast), contrast_img)

                    # --- 5. Gaussian Blur ---
                    if random.random() < 0.5:  # Apply blur augmentation randomly
                        blur_img = cv2.GaussianBlur(face_crop, (5, 5), 0)
                        mp_blur = mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
                        blur_results = face_landmarker.detect(mp_blur)
                        if blur_results and blur_results.face_landmarks:
                            lm_list_blur = blur_results.face_landmarks[0]
                            lm_coords_blur = np.array([[lm.x, lm.y, lm.z] for lm in lm_list_blur]).flatten()
                            lm_coords_blur = normalize(lm_coords_blur.reshape(1, -1)).flatten()
                            blob_blur = cv2.dnn.blobFromImage(blur_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                            face_net.setInput(blob_blur)
                            dnn_out_blur = face_net.forward().flatten()
                            combined_blur = np.hstack((lm_coords_blur, dnn_out_blur))
                            if combined_blur.shape[0] < 4096:
                                combined_blur = np.pad(combined_blur, (0, 4096 - combined_blur.shape[0]), 'constant')
                            else:
                                combined_blur = combined_blur[:4096]
                            features_list.append(combined_blur)
                            labels_list.append(1 if label == "real" else 0)
                            img_filename_blur = f"{label}_{uuid.uuid4().hex}_blur.jpg"
                            cv2.imwrite(os.path.join(img_dir, img_filename_blur), blur_img)

                    # --- 6. Additive Gaussian Noise ---
                    if random.random() < 0.5:  # Apply noise augmentation randomly
                        noise = np.random.normal(0, 10, face_crop.shape).astype(np.uint8)
                        noisy_img = cv2.add(face_crop, noise)
                        mp_noise = mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
                        noise_results = face_landmarker.detect(mp_noise)
                        if noise_results and noise_results.face_landmarks:
                            lm_list_noise = noise_results.face_landmarks[0]
                            lm_coords_noise = np.array([[lm.x, lm.y, lm.z] for lm in lm_list_noise]).flatten()
                            lm_coords_noise = normalize(lm_coords_noise.reshape(1, -1)).flatten()
                            blob_noise = cv2.dnn.blobFromImage(noisy_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                            face_net.setInput(blob_noise)
                            dnn_out_noise = face_net.forward().flatten()
                            combined_noise = np.hstack((lm_coords_noise, dnn_out_noise))
                            if combined_noise.shape[0] < 4096:
                                combined_noise = np.pad(combined_noise, (0, 4096 - combined_noise.shape[0]), 'constant')
                            else:
                                combined_noise = combined_noise[:4096]
                            features_list.append(combined_noise)
                            labels_list.append(1 if label == "real" else 0)
                            img_filename_noise = f"{label}_{uuid.uuid4().hex}_noise.jpg"
                            cv2.imwrite(os.path.join(img_dir, img_filename_noise), noisy_img)
        frame_idx += 1
    cap.release()
    return video_path, features_list, labels_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None,
                        help="Total samples to process (split equally). If not set, process all.")
    parser.add_argument("--threads", type=int, default=16, help="Number of parallel threads.")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run (skip processed videos).")
    parser.add_argument("--frame_step", type=int, default=1, help="Process every Nth frame (default=1).")
    args = parser.parse_args()
    total_count = args.count
    num_threads = args.threads
    resume = args.resume
    frame_step = args.frame_step

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REAL_IMG_DIR, exist_ok=True)
    os.makedirs(FAKE_IMG_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, "processed_videos.log")
    processed_videos = set()
    if resume and os.path.exists(log_path):
        with open(log_path, "r") as lf:
            for line in lf:
                processed_videos.add(line.strip())
        logging.info(f"Resuming. Skipping {len(processed_videos)} already processed videos.")
    elif not resume and os.path.exists(log_path):
        os.remove(log_path)  # fresh start

    # List video files
    real_videos = sorted([os.path.join(REAL_VID_DIR, f) 
                          for f in os.listdir(REAL_VID_DIR) 
                          if f.lower().endswith((".mp4", ".avi", ".mov"))])
    fake_videos = sorted([os.path.join(FAKE_VID_DIR, f) 
                          for f in os.listdir(FAKE_VID_DIR) 
                          if f.lower().endswith((".mp4", ".avi", ".mov"))])
    if resume:
        real_videos = [v for v in real_videos if v not in processed_videos]
        fake_videos = [v for v in fake_videos if v not in processed_videos]

    target_per_class = None
    if total_count:
        target_per_class = total_count // 2
        logging.info(f"Target samples per class: {target_per_class}")

    batch_size = 1000
    batch_features = []
    batch_labels = []
    batch_index = 0
    count_real = 0
    count_fake = 0

    logging.info(f"Found {len(real_videos)} real and {len(fake_videos)} fake videos.")
    # Open log file for appending
    with open(log_path, "a") as log_f:
        # Process real videos
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures_real = {executor.submit(process_video, vid, "real", frame_step): vid 
                            for vid in real_videos 
                            if not (target_per_class and count_real >= target_per_class)}
            for future in concurrent.futures.as_completed(futures_real):
                vid_path, feats, labs = future.result()
                count_real += len(labs)
                batch_features.extend(feats)
                batch_labels.extend(labs)
                log_f.write(f"{vid_path}\n"); log_f.flush()
                if len(batch_features) >= batch_size:
                    npz_path = os.path.join(OUTPUT_DIR, f"data_batch_{batch_index}.npz")
                    np.savez_compressed(npz_path,
                                         features=np.array(batch_features, dtype=np.float32),
                                         labels=np.array(batch_labels, dtype=np.int32))
                    logging.info(f"Saved {len(batch_features)} samples to {npz_path}")
                    batch_features.clear(); batch_labels.clear()
                    batch_index += 1
        if target_per_class and count_real < target_per_class:
            logging.warning(f"Real videos yielded {count_real} samples, adjusting target to {count_real}.")
            target_per_class = count_real
        # Process fake videos
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures_fake = {executor.submit(process_video, vid, "fake", frame_step): vid 
                            for vid in fake_videos 
                            if not (target_per_class and count_fake >= target_per_class)}
            for future in concurrent.futures.as_completed(futures_fake):
                vid_path, feats, labs = future.result()
                count_fake += len(labs)
                batch_features.extend(feats)
                batch_labels.extend(labs)
                log_f.write(f"{vid_path}\n"); log_f.flush()
                if len(batch_features) >= batch_size:
                    npz_path = os.path.join(OUTPUT_DIR, f"data_batch_{batch_index}.npz")
                    np.savez_compressed(npz_path,
                                         features=np.array(batch_features, dtype=np.float32),
                                         labels=np.array(batch_labels, dtype=np.int32))
                    logging.info(f"Saved {len(batch_features)} samples to {npz_path}")
                    batch_features.clear(); batch_labels.clear()
                    batch_index += 1
        # Save remaining data if any
        if batch_features:
            npz_path = os.path.join(OUTPUT_DIR, f"data_batch_{batch_index}.npz")
            np.savez_compressed(npz_path,
                                 features=np.array(batch_features, dtype=np.float32),
                                 labels=np.array(batch_labels, dtype=np.int32))
            logging.info(f"Saved {len(batch_features)} samples to {npz_path}")
            batch_features.clear(); batch_labels.clear()
        logging.info(f"Preprocessing completed. Total samples: real={count_real}, fake={count_fake}")
