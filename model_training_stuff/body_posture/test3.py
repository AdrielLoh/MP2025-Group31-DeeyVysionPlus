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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


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
    
# === Step 1.2: Get and Plot Stats ===
def get_plot_keypoints(tracker, save_location="static/results/body_posture"):
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


import pickle
with open("temp.pkl", "rb") as f:
    keypoints_tracker = pickle.load(f)
get_plot_keypoints(keypoints_tracker)