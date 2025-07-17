import cv2
import numpy as np
import mediapipe as mp
import sys
import os

# Your ROI_INDICES and ROI_COLORS as before...
ROI_INDICES = {
    'left_cheek': [207, 216, 206, 203, 129, 209, 126, 47, 121,
                   120, 119, 118, 117, 111, 116, 123, 147, 187],
    'right_cheek': [350, 277, 355, 429, 279, 331, 423, 426, 436, 427,
                    411, 376, 352, 345, 340, 346, 347, 348, 349],
    'forehead':    [10, 338, 297, 332, 284, 251, 301, 300, 293, 
                    334, 296, 336, 9, 107, 66, 105, 63, 70, 71, 21, 
                    54, 103, 67, 109],
    'chin':        [43, 202, 210, 169, 150, 149, 176, 148, 152, 377, 400, 378, 379,
                    394, 430, 422, 273, 335, 406, 313, 18, 83, 182, 106],
    'nose':        [168, 122, 174, 198, 49, 48, 115, 220, 44, 1, 274, 440,
                    344, 279, 429, 399, 351]
}

ROI_COLORS = {
    'left_cheek':  (0, 255, 0),
    'right_cheek': (0, 128, 255),
    'forehead':    (255, 0, 128),
    'chin':        (255, 255, 0),
    'nose':        (255, 0, 0)
}

def process_image(img_path, out_path, face_mesh):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return False

    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print(f"No face landmarks detected: {os.path.basename(img_path)}")
        return False

    h, w = img.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark

    for roi, indices in ROI_INDICES.items():
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                lx, ly = int(lm.x * w), int(lm.y * h)
                pts.append([lx, ly])
        pts = np.array(pts, dtype=np.int32)
        color = ROI_COLORS[roi]
        # Draw filled polygon with transparency
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        # Draw border
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        # Put label
        centroid = np.mean(pts, axis=0).astype(int)
        cv2.putText(img, roi, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize-rois-batch.py input_folder output_folder")
        sys.exit(1)

    in_folder = sys.argv[1]
    out_folder = sys.argv[2]
    os.makedirs(out_folder, exist_ok=True)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(in_folder) if f.lower().endswith(image_exts)]

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
        for fname in images:
            in_path = os.path.join(in_folder, fname)
            out_path = os.path.join(out_folder, fname)
            ok = process_image(in_path, out_path, face_mesh)
            print(f"{fname}: {'OK' if ok else 'No face found'}")

    print(f"\nDone! Visualized images saved in: {out_folder}")
