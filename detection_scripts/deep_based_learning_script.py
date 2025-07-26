import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.api.models import load_model
import json

# Load the pre-trained deepfake detection model
model = load_model('models/pattern_recognition.keras')

# Load the face detection model
face_net = cv2.dnn.readNetFromCaffe('models/weights-prototxt.txt', 'models/res_ssd_300Dim.caffeModel')

def preprocess_frame(frame):
    if frame.size == 0:
        print("Warning: Trying to resize an empty frame.")
        return None
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

def detect_faces_dnn(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold to filter weak detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

###### This live detection is already redundant for new website, but can leave it for future reference ######
def live_detection(output_folder):
    cap = cv2.VideoCapture(0)  # 0 for default camera
    real_frame_count = 0
    fake_frame_count = 0
    score_list = []  # To store confidence scores

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame read from camera.")
            break

        faces = detect_faces_dnn(frame)
        if not faces:
            print("No faces detected.")
            continue

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                print(f"Empty face region detected: x={x}, y={y}, w={w}, h={h}")
                continue

            preprocessed_face = preprocess_frame(face)
            if preprocessed_face is None:
                continue

            prediction = model.predict(preprocessed_face)
            score = prediction[0][0]
            score_list.append(score)

            if score > 0.5:
                label = 'Fake'
                fake_frame_count += 1
                color = (0, 0, 255)
            else:
                label = 'Real'
                real_frame_count += 1
                color = (0, 255, 0)

            loss_rate = 1 - score
            text = f'{score:.2f}, {loss_rate:.2f}, {label}'
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Real-time Deepfake Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Determine overall result
    if fake_frame_count > real_frame_count:
        overall_result = "Fake"
    else:
        overall_result = "Real"

    # Generate and save graphs
    plot_and_save_graphs(score_list, real_frame_count, fake_frame_count, output_folder)

    return overall_result, real_frame_count, fake_frame_count

def static_video_detection(video_path, output_folder, unique_tag):
    """Performs deepfake detection on an uploaded video."""
    cache_file = os.path.join(output_folder, "cached_results.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
        return (
            cache["overall_result"],
            cache["real_frame_count"],
            cache["fake_frame_count"],
            cache["rvf_plot"],
            cache["conf_plot"]
        )
    
    cap = cv2.VideoCapture(video_path)
    real_frame_count = 0
    fake_frame_count = 0
    score_list = []  # Store confidence scores

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return "Error", 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        faces = detect_faces_dnn(frame)
        if not faces:
            continue

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            preprocessed_face = preprocess_frame(face)
            if preprocessed_face is None:
                continue

            prediction = model.predict(preprocessed_face)
            score = prediction[0][0]
            score_list.append(score)

            if score > 0.5:
                label = 'Fake'
                fake_frame_count += 1
            else:
                label = 'Real'
                real_frame_count += 1

    cap.release()

    # Determine overall result
    overall_result = "Fake" if fake_frame_count > real_frame_count else "Real"

    # Generate and save graphs
    rvf_plot, conf_plot = plot_and_save_graphs(score_list, real_frame_count, fake_frame_count, output_folder, unique_tag)
    
    # Save results to cache before returning
    os.makedirs(output_folder, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({
            "overall_result": overall_result,
            "real_frame_count": real_frame_count,
            "fake_frame_count": fake_frame_count,
            "rvf_plot": rvf_plot,
            "conf_plot": conf_plot
        }, f)

    return overall_result, real_frame_count, fake_frame_count, rvf_plot, conf_plot


# Function to plot and save individual graphs
def plot_and_save_graphs(score_list, real_count, fake_count, output_folder, unique_tag):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    # Plot "Frames (Real vs Fake)"
    plt.figure(figsize=(8, 3))
    plt.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    plt.title('Frames (Real vs Fake)')
    plt.xlabel('Frame Type')
    plt.ylabel('Count')
    plt.tight_layout()
    real_v_fake_plot_path = os.path.join(output_folder, f'deeplearning_real_vs_fake_{unique_tag}.png')
    plt.savefig(real_v_fake_plot_path)
    plt.clf()

    # Plot "Confidence Score Over Time"
    plt.figure(figsize=(8, 3))
    plt.plot(score_list, color='blue')
    plt.title('Confidence Score Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Confidence Score')
    plt.tight_layout()
    conf_plot_path = os.path.join(output_folder, f'deeplearning_confidence_score_{unique_tag}.png')
    plt.savefig(conf_plot_path)
    plt.clf()
    return real_v_fake_plot_path, conf_plot_path
