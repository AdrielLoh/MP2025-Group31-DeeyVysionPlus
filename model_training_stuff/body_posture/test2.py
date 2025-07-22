import os
import cv2
import numpy as np
import pandas as pd
import openpifpaf
import tensorflow as tf
from scipy.spatial import distance
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Configuration
PROCESSED_FOLDER = "static/processed/"
RESULTS_FOLDER = "static/results/body_posture/"
MODEL_PATH_STATIC = 'models/body_posture_model.pth'
MODEL_PATH_LIVE = 'models/body_posture_live.keras'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Ensure directories exist
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# this tracker tracks people by their keypoints using upper-body joints, if the joints are within max_distance, they are matched as the same person
class PersonTracker:
    def __init__(self, max_distance=30, matching_joints=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        self.tracks = {} # person_id : list of keypoints
        self.track_frames = {} # person_id : list of keypoint frame numbers
        self.last_keypoints = {} # person_id : most recent keypoints
        self.next_id = 0 # next available id for new track
        self.max_distance = max_distance # maximum keypoint distance to be considered a match
        self.matching_joints = matching_joints # joints to use for matching

    def _match_score(self, kp1, kp2): # score how matching two sets of keypoints are
        scores = []
        for joint in self.matching_joints:
            v1, v2 =  kp1[joint][2], kp2[joint][2] # extract visibilities of the joints
            if v1 > 0.5 and v2 > 0.5:
                distance = np.linalg.norm(np.array(kp1[joint][:2]) - np.array(kp2[joint][:2])) # get distance between the two joints
                scores.append(distance)
        return np.mean(scores) if scores else np.inf
    

    def _track_similarity(self, track1, track2): #score how similiar two tracks are
        scores = []

        # get 5 samples from each track
        samples1 = self._get_diverse_samples(track1)
        samples2 = self._get_diverse_samples(track2)

        # compare each sample in track1 with each sample in track2
        for kp1 in samples1:
            for kp2 in samples2:
                score = self._match_score(kp1, kp2)
                if score < np.inf:
                    scores.append(score)
        
        # return mean of scores
        return np.mean(scores) if scores else np.inf
    

    def _get_diverse_samples(self, track, sample_size=5):
        used = set() # used joints
        selected_samples = [] # selected samples

        # find samples with unique joints
        for keypoints in track:
            joints = {joint for joint in self.matching_joints if keypoints[joint][2] > 0.5} # get visible joints that are in joints list
            if not joints.issubset(used): # if joints are not all used, take this keypoints
                selected_samples.append(keypoints)
                used |= joints # add joints to used joints set
            if len(selected_samples) >= sample_size:
                break

        # pad to sample size
        if len(selected_samples) < sample_size:
            for keypoints in track:
                if not any(np.array_equal(keypoints, item) for item in selected_samples):
                    selected_samples.append(keypoints)
                    if len(selected_samples) == sample_size:
                        break
        
        return selected_samples

    
    def update(self, detections, frame_number): # update tracks
        # find matches between existing tracks and new detections
        matches = {} # person_id : matching keypoints
        unmatched = list(detections)
        for person_id, keypoints in self.last_keypoints.items():
            best_score = np.inf
            best_keypoints = None
            for keypoints in unmatched:
                score = self._match_score(keypoints, self.last_keypoints[person_id])
                if score < best_score:
                    best_score = score
                    best_keypoints = keypoints
            if best_score < self.max_distance:
                matches[person_id] = best_keypoints
                for i, keypoints in enumerate(unmatched): # because keypoints are arrays, we cannot directly remove them from a list, we have to compare with array methods
                    if np.array_equal(keypoints, best_keypoints):
                        unmatched.pop(i)
                        break

        # add unmatched detections as new tracks
        for keypoints in unmatched:
            matches[self.next_id] = keypoints
            self.next_id+= 1

        # update tracks with keypoints
        for person_id, keypoints in matches.items():
            self.tracks.setdefault(person_id, []).append(keypoints)
            self.track_frames.setdefault(person_id, []).append(frame_number)
            self.last_keypoints[person_id] = keypoints

        # DEBUG
        # print(f"Matches: {matches}")

        return self.tracks, self.track_frames


    def merge(self, max_time_distance=10):
        merge_map = {} # person_id : [person id 1 to merge with, person2, ... etc]
        person_ids = list(self.tracks.keys()) # get person ids safely

        # iterate through tracks to see if they can be merged
        for i in range(len(person_ids)):
            id1 = person_ids[i]
            for j in range(i+1, len(person_ids)):
                id2 = person_ids[j]

                # skip if already in merge map
                if id1 in merge_map.values() or id2 in merge_map.values():
                    continue

                # check if tracks overlap to merge (I LOVE LOGIC STATEMENTS)
                for frame1, keypoint1 in zip(self.track_frames[id1], self.tracks[id1]):
                    for frame2, keypoint2 in zip(self.track_frames[id2], self.tracks[id2]):
                        if abs(frame1 - frame2) <= max_time_distance:
                            for joint in self.matching_joints:
                                visibility1, visibility2 = keypoint1[joint][2], keypoint2[joint][2]
                                if visibility1 > 0.5 and visibility2 > 0.5:
                                    d = np.linalg.norm(np.array(keypoint1[joint][:2]) - np.array(keypoint2[joint][:2]))

                                    # add to merge map
                                    if d < self.max_distance:
                                        if any(id1 in map_list for map_list in merge_map.values()):
                                            for key, map_list in merge_map.items():
                                                if id1 in map_list:
                                                    merge_map[key].append(id2)
                                                    break
                                        else:
                                            merge_map.setdefault(id1, []).append(id2)
                                        break

                        if any(id1 in map_list for map_list in merge_map.values()) or any(id2 in map_list for map_list in merge_map.values()):
                            break
                    if any(id1 in map_list for map_list in merge_map.values()) or any(id2 in map_list for map_list in merge_map.values()):
                        break

        # merge tracks
        for id1, ids_to_merge in merge_map.items():
            for id2 in ids_to_merge:
                self.tracks[id1].extend(self.tracks[id2])
                self.track_frames[id1].extend(self.track_frames[id2])
                del self.tracks[id2]
                del self.track_frames[id2]
                del self.last_keypoints[id2]

        # remove tracks with less than 5 keypoints
        for pid in list(self.tracks.keys()):
            keypoints = self.tracks[pid]
            if len(keypoints) < 5:
                del self.tracks[pid]
                del self.track_frames[pid]
                del self.last_keypoints[pid]
        
        return self.tracks, self.track_frames

    
# Model Classifier
class PoseClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=256, num_layers=2, num_classes=2):
        super(PoseClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out
    

# Load model
model=PoseClassifier().to(device)
checkpoint = torch.load(MODEL_PATH_STATIC)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
    
# === Step 0.5: Preprocess Video Frames ===
def preprocess_frame(frame):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Gamma correction
    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, look_up_table)

    # White balance correction (OpenCV’s xphoto module)
    wb = cv2.xphoto.createSimpleWB()
    white_balanced = wb.balanceWhite(gamma_corrected)

    return white_balanced


# === Step 1: Extract and Track Keypoints from Video ===
def extract_keypoints(video_path, show_process=True):
    predictor = openpifpaf.Predictor(checkpoint='resnet50')
    cap = cv2.VideoCapture(video_path)
    print("Opening:", video_path + " Successful:", os.path.exists(video_path))
    if not cap.isOpened():
        print("Failed to open video")
    tracker = PersonTracker()
    frame_num = 0

    while cap.isOpened():
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame = preprocess_frame(frame)

        # Get keypoints
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB for OpenPifPaf
        predictions, _, _ = predictor.numpy_image(rgb_frame) # get only predictions
        
        # Process predictions
        detections = []
        for annotation in predictions:
            keypoints = annotation.data # np array shape: (17, 3), COCO keypoints format
            detections.append(keypoints)

        # Update tracker with new detections
        tracker.update(detections, frame_num)

        if show_process:
            for keypoints in detections:
                for i, (x, y, v) in enumerate(keypoints):
                    if v > 0.5:  # Only plot visible keypoints
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Optionally draw limbs (connect keypoints)
                # Example for some COCO keypoint connections:
                connections = [
                    (5, 7), (7, 9),     # Right Arm
                    (6, 8), (8, 10),    # Left Arm
                    (11, 13), (13, 15), # Right Leg
                    (12, 14), (14, 16), # Left Leg
                    (5, 6), (11, 12),   # Shoulders and Hips
                ]
                for idx1, idx2 in connections:
                    if keypoints[idx1][2] > 0.5 and keypoints[idx2][2] > 0.5:
                        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

            cv2.imshow("OpenPifPaf Pose Annotation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1

    cap.release()
    if show_process:
        cv2.destroyAllWindows()
    return tracker


# === Step 1.1: Plot Keypoints on Video ===
def plot_keypoints_video(video_path, keypoints_dict, frames, results_folder=RESULTS_FOLDER):
    # prepare video + directory
    filename = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(results_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # prepare writers and frame to keypoint map
    writers = {}
    frame_keypoint_map = {} # frame_number : {person_id: keypoints}
    for person_id in keypoints_dict:
        # create writers for each person
        output_path = os.path.join(results_folder, f"{filename}_person_{person_id}.mp4")
        writers[person_id] = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # map keypoints to frames
        person_keypoints = keypoints_dict[person_id]
        person_frames = frames[person_id]
        for frame_id, keypoints in zip(person_frames, person_keypoints):
            frame_keypoint_map.setdefault(frame_id, {})[person_id] = keypoints
    
    # process video frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # set to first frame
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num in frame_keypoint_map:
            for person_id, keypoints in frame_keypoint_map[frame_num].items():
                person_frame = frame.copy() # copy current frame for this person

                # draw main keypoints
                for (x, y, v) in keypoints:
                    if v > 0.5:
                        cv2.circle(person_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                # draw limb connections
                connections = [
                    (5, 7), (7, 9),     # Right Arm
                    (6, 8), (8, 10),    # Left Arm
                    (11, 13), (13, 15), # Right Leg
                    (12, 14), (14, 16), # Left Leg
                    (5, 6), (11, 12),   # Shoulders and Hips
                ]
                for i, j in connections:
                    if keypoints[i][2] > 0.5 and keypoints[j][2] > 0.5:
                        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                        cv2.line(person_frame, pt1, pt2, (255, 0, 0), 2)

                # label the person (optional)
                visible_pts = [kp[:2] for kp in keypoints if kp[2] > 0.5]
                if visible_pts:
                    cx, cy = np.mean(visible_pts, axis=0)
                    cv2.putText(person_frame, f"ID {person_id}", (int(cx), int(cy)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                # write to video
                writers[person_id].write(person_frame)
            frame_num += 1
    
    # finish and cleanup
    cap.release()
    for writer in writers.values():
        writer.release()
    print(f"Keypoint videos saved to {results_folder}")


# === Step 1.2: Get and Plot Stats ===
def get_plot_keypoints(tracker, save_location=RESULTS_FOLDER):
    tracker.visible_joints = {} # person_id : list of visible joints
    tracker.joint_visiblity = {} # person_id : list of average visibility of each joint (percentage of frames where joint is visible)
    tracker.joint_confidence = {} # person_id : list of average confidence of each joint
    tracker.joint_instability = {} # person_id : list of instability of each visible joint (std dev)
    tracker.track_lengths = {} # person_id : length of track

    # calculate per person
    for person_id, keypoints in tracker.tracks.items():
        tracker.track_lengths[person_id] = len(keypoints)

        # get joint visiblity and confidences per frame
        visible_joints = {} # joint idx : list of visible joint positons
        joint_confidences = {} # joint idx : list of confidences
        for keypoint in keypoints:
            for joint in range(17):
                if keypoint[joint][2] > 0.5:
                    visible_joints.setdefault(joint, []).append(keypoint[joint][:2])
                joint_confidences.setdefault(joint, []).append(keypoint[joint][2])
        
        # calculate stats
        tracker.visible_joints[person_id] = list(visible_joints.keys())
        tracker.joint_visiblity[person_id] = []
        tracker.joint_confidence[person_id] = []
        tracker.joint_instability[person_id] = []
        for joint in range(17):
            joint_vis = visible_joints.get(joint, [])
            if joint_vis:
                tracker.joint_visiblity[person_id].append(len(joint_vis) / len(keypoints))
                tracker.joint_confidence[person_id].append((sum(joint_confidences[joint]) / len(joint_confidences[joint])))
                coordinates = np.array(visible_joints[joint])
                tracker.joint_instability[person_id].append(np.sqrt(np.mean(np.var(coordinates))))
            else:
                tracker.joint_visiblity[person_id].append(0)
                tracker.joint_confidence[person_id].append(0)
                tracker.joint_instability[person_id].append(0)

        # plot person graphs
        joint_names = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder',
                    'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee',
                    'RKnee', 'LAnkle', 'RAnkle']
        plt.figure(figsize=(14, 5))

        # joint visibility
        plt.subplot(1, 3, 1)
        plt.bar(joint_names, tracker.joint_visiblity[person_id])
        plt.title("Joint Visibility")
        plt.xticks(rotation=45)


        # confidence per joint
        plt.subplot(1, 3, 2)
        plt.bar(joint_names, tracker.joint_confidence[person_id])
        plt.title("Joint Confidence")
        plt.xticks(rotation=45)


        # stability per joint
        plt.subplot(1, 3, 3)
        plt.bar(joint_names, tracker.joint_instability[person_id])
        plt.title("Joint Stability")
        plt.xticks(rotation=45)

        # save plots
        plt.tight_layout()
        plt.savefig(os.path.join(save_location, f'person_{person_id}.png'))
        plt.clf()
    
    # plot cross-person graphs
    person_ids = list(tracker.visible_joints.keys())
    visible_joints = [len(tracker.visible_joints[person_id]) for person_id in person_ids]
    track_lengths = [tracker.track_lengths[person_id] for person_id in person_ids]
    plt.figure(figsize=(12, 5))

    # Average visible joints
    plt.subplot(1, 2, 1)
    plt.bar(person_ids, visible_joints, color="skyblue")
    plt.xticks(person_ids, [f"Person {pid}" for pid in person_ids], rotation=45)
    plt.title("Avg Visible Joints Per Person")
    plt.ylabel("Joints")
    plt.xlabel("Person ID")

    # Track duration
    plt.subplot(1, 2, 2)
    plt.bar(person_ids, track_lengths, color="salmon")
    plt.xticks(person_ids, [f"Person {pid}" for pid in person_ids], rotation=45)
    plt.title("Track Length Per Person")
    plt.ylabel("Frames")
    plt.xlabel("Person ID")

    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f'overall.png'))
    plt.clf()

    return tracker


# === Step 2: Normalize Keypoints (Center on pelvis, scale by torso length) ===
def normalize_keypoints(keypoints_seq, drop_visibility=True): # keypoints_seq: list of [ (x, y, v), ... ] per frame
    # Normalize each frame: center on pelvis (midpoint of hips), scale by torso length (shoulder to hip)
    norm_seq = []
    for frame in keypoints_seq:
        kp = np.array(frame) # get keypoints as numpy array
        if kp.shape[0] < 8:  # Not enough keypoints
            norm_seq.append(kp)
            continue

        # Math to find torso length
        # COCO keypoint format: 11=left hip, 12=right hip, 5=left shoulder, 6=right shoulder
        pelvis = np.mean([kp[11][:2], kp[12][:2]], axis=0)
        shoulder = np.mean([kp[5][:2], kp[6][:2]], axis=0)
        torso_length = np.linalg.norm(shoulder - pelvis) # get distance between shoulders and pelvis
        if torso_length < 1e-3: # avoid division by zero
            torso_length = 1.0
        normed = kp.copy()
        normed[:,0:2] = (kp[:,0:2] - pelvis) / torso_length # subtract pelvis position to normalise, divide by torso length to scale

        # Flatten frame
        if drop_visibility:
            flat = normed[:, :2].flatten()  # shape: [34]
        else:
            flat = normed.flatten()         # shape: [51]

        norm_seq.append(flat)
    return norm_seq


# == Step 3: Predict Deepfake using Sequence ==
def predict_deepfake(normalized_arr):
    results, confidences = {}, {}
    results_arr = {
        0 : "Real",
        1 : "Fake"
    }
    for person_id, seq in normalized_arr.items():
        seq = torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(seq)
            probabilities = torch.softmax(output, dim=1)
            confidence, result = torch.max(probabilities, dim=1)

        results[person_id], confidences[person_id] = results_arr[result.item()], confidence.item()

    return {
        "results": results,
        "confidences": confidences,
    }


# === Full Video Processing Pipeline ===
def detect_body_posture(video_path):
    import pickle
    with open("temp.pkl", "rb") as f:
        keypoints_tracker = pickle.load(f)
    # Step 1: Extract Keypoints
    #keypoints_tracker = extract_keypoints(video_path)
    keypoints_dict, frames = keypoints_tracker.merge()
    if keypoints_dict is None:
        return {"error": "Failed to extract keypoints"}
    
    # Step 1.1: Plot Keypoints on Video
    # plot_keypoints_video(video_path, keypoints_dict, frames)

    # Step 1.2: Retrieve Keypoint Stats
    get_plot_keypoints(keypoints_tracker)

    # Step 2: Preprocess Extracted Keypoints    
    normalized_dict = {}
    for person_id, seq in keypoints_dict.items():
        norm_seq = normalize_keypoints(seq)
        normalized_dict[person_id] = norm_seq 
    if not normalized_dict:
        return {"error": "Failed to normalize keypoints"}

    # Step 3: Run Prediction
    prediction_results = predict_deepfake(normalized_dict)

    # Step 4: Return Results
    results = []
    for person_id in keypoints_tracker.tracks.keys():
        result = {
            "person_id" : person_id,
            "result" : prediction_results["results"][person_id],
            "result_confidence" : prediction_results["confidences"][person_id],
            "track_length" : keypoints_tracker.track_lengths[person_id],
            "joint_visiblity" : keypoints_tracker.joint_visiblity[person_id],
            "joint_confidence" : keypoints_tracker.joint_confidence[person_id],
            "joint_instability" : keypoints_tracker.joint_instability[person_id]
        }
        results.append(result)
    return results