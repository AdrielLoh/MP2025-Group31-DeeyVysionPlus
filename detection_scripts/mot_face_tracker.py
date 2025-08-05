import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

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