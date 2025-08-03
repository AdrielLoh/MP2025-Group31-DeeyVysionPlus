import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from multiprocessing import Pool
from tqdm import tqdm
import random

# directories for storage
FAKE_VID_DIR = "/mnt/d/fake_videos"
REAL_VID_DIR = "/mnt/d/real_videos"
OUTPUT_DIR = "/mnt/d/preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_VIDEOS = None #max number of videos can be set to limit number of videos preprocessed. None = all videos preprocessed
FRAME_INTERVAL = 0.5 #seconds between frames ot process. 0.5 is set to process every 15th frame in a 30 fps video
FACE_SIZE = 64 #resizing detected faces to 64x64 pixels
AUGMENT_PROB = 0.5  # 50% chance for the frame to be augmented

# Load OpenCV DNN face detector 
PROTO_PATH = "models/weights-prototxt.txt"  # path to deploy.prototxt file
MODEL_PATH = "models/res_ssd_300Dim.caffeModel"    # path to caffemodel
if not os.path.isfile(PROTO_PATH) or not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("Face detector model files not found. Please download deploy.prototxt and .caffemodel.")

face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#jpeg compression augmentation to imitate lower resolution videos 
def random_jpeg_compression(image, quality_range=(10, 90)):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(*quality_range)] #random compression levels
    _, encimg = cv2.imencode('.jpg', image, encode_param) #compress in memory and encodes it into jpg image
    return cv2.imdecode(encimg, 1) #decode as a BGR image

#gaussian blur to soften image and reduce sharp details only found in high quality videos
#imitates camera motion/ focus loss
def random_gaussian_blur(image, sigma_range=(0.5, 1.5)):
    if random.random() < 0.5:
        sigma = random.uniform(*sigma_range) #higher sigma = more blur
        ksize = int(2 * round(sigma * 2) + 1) #kernel size is set according to sigma value and is always odd by adding 1, centering the kernel (3x3, 5x5, 9x9)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return image

#random brightness adjustments to make the model focus on facial structure and ignore lighting conditions 
def random_brightness_contrast(image, brightness_range=(-32, 32), contrast_range=(0.5, 1.5)):
    if random.random() < 0.5:
        brightness = random.randint(*brightness_range) #dimming/ brightening of pixels
        contrast = random.uniform(*contrast_range) #amplifying dark/ bright parts of the face
        image = image.astype(np.float32) * contrast + brightness #convert pixel data to 32 bit float as some pixel values may exceed 8 bit range
        image = np.clip(image, 0, 255).astype(np.uint8) #limiting pixel values to stay between 0 and 255 (limit for rgb channel) and convert back to usual 8-bit image for compatibility and saving memory
    return image

#random gaussian noise adds random static to image
def random_noise(image, mean=0, stddev=10):
    if random.random() < 0.5:
        noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        noisy_img = image.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img
    return image

#random blacking out a region of the image
def apply_cutout(image, size=12):
    if random.random() <0.5:
        h, w = image.shape[:2]  # get height and width (works for both grayscale and color)
        top = np.random.randint(0, h - size) #random vertical coordinate to place the top edge of the cutout square
        left = np.random.randint(0, w - size) #random horizontal coordinate to place the left edge of the cutout square
        if image.ndim == 2:  # check if image is a grayscale image (2D)
            image[top:top+size, left:left+size] = 0
        else:  # checking if image is color image (3D)
            image[top:top+size, left:left+size, :] = 0 #sets all pixel values to 0 = set all pixel values within the cutout to black
        return image


def apply_random_augmentations(image):
    # applies multiple augmentations in random order
    aug_image = image.copy()
    funcs = [
        apply_cutout,
        random_jpeg_compression,
        random_gaussian_blur,
        random_brightness_contrast,
        random_noise
    ]
    random.shuffle(funcs) #shuffle the augmentation functions for every image
    for func in funcs:
        aug_image = func(aug_image)
    return aug_image

#extracting HOG from faces 
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

def process_video(video_path, label, batch_index):
    cap = cv2.VideoCapture(video_path) #opening video to read it
    if not cap.isOpened():
        print(f"Warning: unable to open video {video_path}")
        return None #skip if the video cannot be read
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 #obtain frame rate of video or default to 30 fps otherwise 
    frame_interval_frames = int(round(fps * FRAME_INTERVAL)) #calculate the nth frame to be processed
    features_list = [] #array to store extracted features
    frame_count = 0 #current frame which is 0 by default
    success = True 
    while success:
        success = cap.grab() #move to next frame
        if not success:
            break #if not possible it is the end of the video file
        if frame_count % frame_interval_frames == 0: #processing every nth frame based on previous calculation of frames to be skipped
            ret, frame = cap.retrieve() #read the corresponding frame and decode it
            if not ret or frame is None:
                frame_count += 1 #skip frame if it cannot be read
                continue
            (h, w) = frame.shape[:2] #obtain frame dimensions
            #prep the frame for face detection by standardizing it for the face detector caffemodel
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         (104.0, 117.0, 123.0), swapRB=False, crop=False)
            face_net.setInput(blob)
            detections = face_net.forward()
            best_conf = 0 #track highest confidence frame detected in the frame
            best_box = None #track location of it
            for i in range(detections.shape[2]): #loop over all detected faces
                conf = detections[0, 0, i, 2]
                if conf > 0.5 and conf > best_conf:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) #scale bounding box size to frame size
                    (x1, y1, x2, y2) = box.astype('int') #convert box to pixel values
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 - x1 > 0 and y2 - y1 > 0: #only keeping valid boxes
                        best_conf = conf
                        best_box = (x1, y1, x2, y2) 
            if best_box is None: #skip frame if no face has high confidence rate
                frame_count += 1
                continue
            (x1, y1, x2, y2) = best_box #coordinates of best face bounding box
            face = frame[y1:y2, x1:x2] #cropping the face out of the frame
            face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE)) #resizing to 64x64 pixels
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            if np.random.rand() < AUGMENT_PROB:
                # apply augmentations if random number generation is < 0.5
                # Convert back to 3 channels for some augmentations, then to gray again for feature extraction
                face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
                face_rgb = apply_random_augmentations(face_rgb)
                face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2GRAY)
            hog_vec = extract_hog_features(face_gray)
            lbp_vec = extract_lbp_features(face_gray)
            #combine extracted HOG and LBP into one feature vector
            feat_vec = np.concatenate([hog_vec, lbp_vec]).astype('float32')
            if np.var(feat_vec) < 1e-6: #skip frame if extracted features are not useful
                frame_count += 1
                continue
            features_list.append(feat_vec)
        frame_count += 1
    cap.release() #release frame to save memory and resources
    if not features_list:
        return None #skip if no features
    features_arr = np.stack(features_list, axis=0)
    labels_arr = np.full((features_arr.shape[0],), label, dtype='uint8') #array of labels for the batch
    cls = 'fake' if label == 1 else 'real' #naming of batches fake or real if video has label 1
    out_path = os.path.join(OUTPUT_DIR, f"{cls}_batch_{batch_index:04d}.npz") 
    np.savez_compressed(out_path, features=features_arr, labels=labels_arr) #saving it as npz file
    return out_path

#pass multiple arguments to enable multiprocessing
def process_video_wrapper(args):
    return process_video(*args)

# main execution
if __name__ == "__main__":
    fake_videos = sorted([os.path.join(FAKE_VID_DIR, f) for f in os.listdir(FAKE_VID_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
    real_videos = sorted([os.path.join(REAL_VID_DIR, f) for f in os.listdir(REAL_VID_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
    total_fakes = len(fake_videos)
    total_reals = len(real_videos)
    if MAX_VIDEOS is not None:
        per_class = MAX_VIDEOS // 2 #balancing videos if MAX_VIDEOS is defined
        total_fakes = min(total_fakes, per_class)
        total_reals = min(total_reals, per_class)
    else:
        min_count = min(total_fakes, total_reals) #if not defined take the maxiumum number of the class with lesser videos
        total_fakes = total_reals = min_count
    fake_videos = fake_videos[:total_fakes]
    real_videos = real_videos[:total_reals]
    print(f"Processing {len(fake_videos)} fake videos and {len(real_videos)} real videos...")

#storing preprocessed data in npz files for smaller size and fixed naming system to differentiate deepfake and original data (mainly for readability)
    tasks = []
    batch_index = 0
    for vid in fake_videos:
        out_path = os.path.join(OUTPUT_DIR, f"fake_batch_{batch_index:04d}.npz")
        if os.path.exists(out_path):
            batch_index += 1 #skip if batch file already exists
            continue
        tasks.append((vid, 1, batch_index)) #label the video 1 if it is fake 
        batch_index += 1
    for vid in real_videos:
        out_path = os.path.join(OUTPUT_DIR, f"real_batch_{batch_index:04d}.npz")
        if os.path.exists(out_path):
            batch_index += 1
            continue
        tasks.append((vid, 0, batch_index)) #label the video 0 if it is real
        batch_index += 1

    num_workers = 8 #use 8 threads for parallel processing
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_video_wrapper, tasks),
                      total=len(tasks), desc="Preprocessing Videos"):
            pass

    print("Preprocessing completed. Outputs saved to", OUTPUT_DIR)