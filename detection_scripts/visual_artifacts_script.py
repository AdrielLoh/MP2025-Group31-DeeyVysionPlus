import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog, local_binary_pattern
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR) #block logging spam
import itertools
import subprocess
import json
from detection_scripts.mot_face_tracker import robust_track_faces

MODEL_PATH = "models/visual_artifacts_model.keras"
PROTO_PATH = "models/weights-prototxt.txt"
MODEL_CAFFE_PATH = "models/res_ssd_300Dim.caffeModel"
FACE_SIZE = 64
MIN_FRAMES = 30 
#default output directory for results
DEFAULT_OUTPUT_DIR = "static/results" 

print(f"[DEBUG] Loading detection model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[DEBUG] Loading face detector: {PROTO_PATH}, {MODEL_CAFFE_PATH}")
face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_CAFFE_PATH)

#same extraction functions as preprocessing script
def extract_hog_features(img_gray):
    return hog(img_gray, orientations=9, pixels_per_cell=(8, 8), #9 possible gradient directions with 8x8 pixel size per cell
               cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True) #each block = 2x2 cells and output results as single feature vector

#extracting LBP from faces
def extract_lbp_features(img_gray):
    #small scale lbp code extraction for every pixel for capturing finer details
    lbp_r1 = local_binary_pattern(img_gray, P=8, R=1, method='uniform') #each pixel gets an lbp code after being compared to its 8 neighboring pixels within 1 pixel radius
    #larger scale lbp code extraction for every pixel for capturing larger details
    lbp_r2 = local_binary_pattern(img_gray, P=16, R=2, method='uniform')  #each pixel gets an lbp code after being compared to its 16 neighboring pixels within 2 pixel radius
    #np.histogram counts frequency for each lbp code and puts them into a histogram
    hist_r1, _ = np.histogram(lbp_r1, bins=np.arange(0, 8+2), range=(0, 8+1)) #10 bins for the histogram. 0 to 8 is for uniform patterns (transition from 0 to 1/ 1 to 0) while 9 is for non-uniform patterns where there are multiple transitions
    hist_r2, _ = np.histogram(lbp_r2, bins=np.arange(0, 16+2), range=(0, 16+1)) #18 bins for the histogram with the same rationale as the previous histogram
    #convert both histograms to  float32 for later usage
    hist_r1 = hist_r1.astype('float32')
    hist_r2 = hist_r2.astype('float32')
    #ensure that maximum sum of the histogram is 1
    if hist_r1.sum() != 0:
        hist_r1 /= hist_r1.sum()
    if hist_r2.sum() != 0:
        hist_r2 /= hist_r2.sum()
    return np.concatenate([hist_r1, hist_r2]) #combine both histograms into one feature vector

def run_visual_artifacts_detection(*args, **kwargs):
    #get video path from function arguments
    if len(args) > 0:
        video_path = args[0]
    else:
        video_path = kwargs.get('video_path')
    if video_path is None:
        raise ValueError("No video_path provided!")

    if 'output_dir' in kwargs:
        output_dir = kwargs['output_dir']
    elif len(args) > 1:
        output_dir = args[1]
    else:
        output_dir = "./static/artifact_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # caching results into json file
    cache_file = os.path.join(output_dir, "cached_results.json")
    #if video was already processed and results were returned, cached results are fetched and returned
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
        return cache

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "reason": f"Cannot open video: {video_path}"}

    #get video fps or default to 30
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_frames = 1
    frame_count = 0
    all_frame_boxes = []
    all_face_features = []
    frames = []

    while True:
        success = cap.grab()
        if not success:
            break
        if frame_count % frame_interval_frames == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                frame_count += 1
                continue
            (h, w) = frame.shape[:2]
            #prepare frame for face detection by the caffe model
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         (104.0, 117.0, 123.0), swapRB=False, crop=False)
            face_net.setInput(blob)
            detections = face_net.forward()
            frame_boxes = []
            face_feats = []
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.6:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype('int')
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    bw, bh = x2 - x1, y2 - y1
                    if bw > 32 and bh > 32:   # filter out bounding boxes that are too small to minimize false positive face detections
                        frame_boxes.append([x1, y1, bw, bh])
                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            face_feats.append(None)
                            continue
                        face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
                        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                        hog_vec = extract_hog_features(face_gray)
                        lbp_vec = extract_lbp_features(face_gray)
                        #combine extracted HOG and LBP features 
                        feat_vec = np.concatenate([hog_vec, lbp_vec]).astype('float32')
                        #skip face if features are invalid
                        if np.var(feat_vec) < 1e-6 or np.any(np.isnan(feat_vec)):
                            face_feats.append(None)
                            continue
                        face_feats.append(feat_vec)
            all_frame_boxes.append(frame_boxes)
            all_face_features.append(face_feats)
            frames.append(frame.copy())
        frame_count += 1
    cap.release()

    if len(frames) == 0 or len(all_frame_boxes) == 0:
        return {"success": False, "reason": "No faces detected in the video."}

    tracks = robust_track_faces(all_frame_boxes, frames)

    # for each track, collect the frame indices, bounding boxes, predictions, and features
    per_face_data = {}
    for tid, track in tracks.items():
        per_face_data[tid] = {"frame_indices": [], "boxes": [], "probs": [], "features": []}
        for (frame_idx, box) in track:
            # Find detection index in frame
            box_idx = None
            for b_idx, b in enumerate(all_frame_boxes[frame_idx]):
                if np.allclose(box, b, atol=3):
                    box_idx = b_idx
                    break
            if box_idx is not None and all_face_features[frame_idx][box_idx] is not None:
                feat_vec = all_face_features[frame_idx][box_idx]
                prob = float(model.predict(np.expand_dims(feat_vec, axis=0), verbose=0)[0,0])
                per_face_data[tid]["frame_indices"].append(frame_idx)
                per_face_data[tid]["boxes"].append(box)
                per_face_data[tid]["probs"].append(prob)
                per_face_data[tid]["features"].append(feat_vec)
    #keep track with enough frames
    per_face_data = {tid: d for tid, d in per_face_data.items() if len(d["probs"]) >= MIN_FRAMES}

    # creating video with all faces/IDs overlay which is shown as bounding boxes around the face
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_temp = os.path.join(output_dir, "visual_artifact_overlay.mp4").replace("\\", "/")
    writer = cv2.VideoWriter(video_temp, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for i, frame in enumerate(frames):
        frame_draw = frame.copy()
        for tid, d in per_face_data.items():
            if i in d["frame_indices"]:
                idx = d["frame_indices"].index(i)
                box = d["boxes"][idx]
                prob = d["probs"][idx]
                label = "Fake" if prob > 0.5 else "Real"
                color = (0,255,0) if prob < 0.5 else (0,0,255)
                x1, y1, w, h = [int(x) for x in box]
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_draw, f"ID{tid}-{label}:{prob:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        writer.write(frame_draw)
    writer.release()
    def reencode_mp4_for_html5(input_path, output_path):
        current_wd = os.getcwd()
        input_path_full = os.path.join(current_wd, input_path)
        output_path_full = os.path.join(current_wd, output_path)
        # ffmpeg to reencode with H.264 video and AAC audio for HTML5 compatibility
        cmd = [
            "ffmpeg", "-y", "-i", input_path_full,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path_full
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # after writer.release(), re-encode to guarantee browser compatibility to enable displaying of overlay video 
    h264_overlay = os.path.join(output_dir, "visual_artifact_overlay_h264.mp4")
    reencode_mp4_for_html5(video_temp, h264_overlay)
    # update rel_video_path accordingly
    rel_video_path = os.path.relpath(h264_overlay, "static").replace("\\", "/")
    if os.path.exists(video_temp):
        os.remove(video_temp)
    # analysis for every face and charts expaining the results
    face_results = []
    for tid, d in per_face_data.items():
        if len(d["probs"]) < MIN_FRAMES:
            continue
        avg_prob = np.mean(d["probs"])
        decision = "fake" if avg_prob >= 0.5 else "real"
        num_real = sum(1 for p in d["probs"] if p < 0.5)
        num_fake = sum(1 for p in d["probs"] if p >= 0.5)
        charts = []
        # framewise Probability
        plt.figure(figsize=(10,5))
        plt.plot(d["frame_indices"], d["probs"], marker='o')
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.title(f"Face {tid} - Framewise Fake Probability")
        plt.xlabel("Frame Number")
        plt.ylabel("Predicted Probability (Fake)")
        framewise_path = os.path.join(output_dir, f"face_{tid}_framewise_probabilities.png")
        plt.savefig(framewise_path)
        plt.close()
        charts.append(os.path.relpath(framewise_path, "static").replace("\\", "/"))
        # histogram of fake probabilities
        plt.figure(figsize=(10,5))
        plt.hist(d["probs"], bins=20, color='orange', edgecolor='k')
        plt.axvline(np.mean(d["probs"]), color='b', linestyle='--', label=f"Mean={avg_prob:.2f}")
        plt.axvline(0.5, color='gray', linestyle='--', label="Threshold=0.5")
        plt.title(f"Face {tid} - Histogram of Fake Probabilities")
        plt.xlabel("Fake Probability")
        plt.ylabel("Count")
        plt.legend()
        prob_hist_path = os.path.join(output_dir, f"face_{tid}_probability_histogram.png")
        plt.savefig(prob_hist_path)
        plt.close()
        charts.append(os.path.relpath(prob_hist_path, "static").replace("\\", "/"))
        # real vs fake frame count bar chart
        plt.figure(figsize=(10,5))
        plt.bar(['Fake', 'Real'], [num_fake, num_real], color=['red', 'green'], alpha=0.7)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title(f'Face {tid} - Real vs Fake Frame Counts')
        bar_path = os.path.join(output_dir, f"face_{tid}_real_vs_fake.png")
        plt.savefig(bar_path)
        plt.close()
        charts.append(os.path.relpath(bar_path, "static").replace("\\", "/"))

        # smoothed fake probability, it is for
        window = 5
        if len(d["probs"]) >= window:
            smoothed_probs = np.convolve(d["probs"], np.ones(window)/window, mode='valid')
            plt.figure(figsize=(10,5))
            plt.plot(d["frame_indices"][:len(smoothed_probs)], smoothed_probs, marker='o', color='blue', label='Smoothed')
            plt.axhline(0.5, color='gray', linestyle='--')
            plt.title(f"Face {tid} - Smoothed Fake Probability")
            plt.xlabel("Frame Number")
            plt.ylabel("Fake Probability")
            plt.legend()
            smoothed_path = os.path.join(output_dir, f"face_{tid}_smoothed_probability.png")
            plt.savefig(smoothed_path)
            plt.close()
            charts.append(os.path.relpath(smoothed_path, "static").replace("\\", "/"))
        # CDF
        probs_sorted = np.sort(d["probs"])
        cdf = np.arange(1, len(probs_sorted)+1) / len(probs_sorted)
        plt.figure(figsize=(10,5))
        plt.plot(probs_sorted, cdf, label='CDF')
        plt.xlabel('Fake Probability')
        plt.ylabel('Cumulative Fraction of Frames')
        plt.title(f'Face {tid} - CDF of Fake Probability')
        cdf_path = os.path.join(output_dir, f"face_{tid}_probability_cdf.png")
        plt.savefig(cdf_path)
        plt.close()
        charts.append(os.path.relpath(cdf_path, "static").replace("\\", "/"))

        # prediction variance chart
        if len(d["probs"]) >= window:
            variances = [np.var(d["probs"][max(0,i-window):i+1]) for i in range(len(d["probs"]))]
            plt.figure(figsize=(10,5))
            plt.plot(d["frame_indices"], variances)
            plt.xlabel('Frame Number')
            plt.ylabel('Prediction Variance')
            plt.title(f'Face {tid} - Prediction Variance (Rolling Window)')
            variance_path = os.path.join(output_dir, f"face_{tid}_prediction_variance.png")
            plt.savefig(variance_path)
            plt.close()
            charts.append(os.path.relpath(variance_path, "static").replace("\\", "/"))
        # scatter chart showing prediction results for each frame
        colors = ['red' if p >= 0.5 else 'green' for p in d["probs"]]
        plt.figure(figsize=(10,5))
        plt.scatter(d["frame_indices"], d["probs"], c=colors, alpha=0.7)
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.title(f'Face {tid} - Framewise Fake Probability (Class Color)')
        plt.xlabel('Frame Number')
        plt.ylabel('Fake Probability')
        scatter_path = os.path.join(output_dir, f"face_{tid}_scatter_framewise_probability.png")
        plt.savefig(scatter_path)
        plt.close()
        charts.append(os.path.relpath(scatter_path, "static").replace("\\", "/"))
        # Run Length
        class_seq = [int(p >= 0.5) for p in d["probs"]]
        run_lengths = [len(list(group)) for key, group in itertools.groupby(class_seq)]
        if run_lengths:
            plt.figure(figsize=(10,5))
            plt.hist(run_lengths, bins=range(1, max(run_lengths)+2), align='left', color='purple')
            plt.xlabel('Run Length')
            plt.ylabel('Count')
            plt.title(f'Face {tid} - Consecutive Fake/Real Runs')
            runlen_path = os.path.join(output_dir, f"face_{tid}_run_length_histogram.png")
            plt.savefig(runlen_path)
            plt.close()
            charts.append(os.path.relpath(runlen_path, "static").replace("\\", "/"))

        face_results.append({
            "track_id": tid,
            "result": "Fake" if decision == "fake" else "Real",
            "real_count": num_real,
            "fake_count": num_fake,
            "confidence": round(avg_prob * 100, 2),
            "charts": charts
        })

    # caching of results
    with open(cache_file, "w") as f:
        json.dump({
            "success": True,
            "face_results": face_results,
            "video_with_boxes": rel_video_path
        }, f)

    if not face_results:
        return {"success": False, "reason": "No faces detected in the video."}
    
    return {
        "success": True,
        "face_results": face_results,
        "video_with_boxes": rel_video_path
    }

