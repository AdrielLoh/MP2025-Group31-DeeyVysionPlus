from flask import Flask, render_template, request, jsonify, url_for
import os
import logging
import subprocess
import uuid
from werkzeug.utils import secure_filename
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import multiprocessing
from waitress import serve

logging.basicConfig(level=logging.DEBUG)
from detection_scripts.deep_based_learning_script import live_detection as deep_learning_live_detection
from detection_scripts.deep_based_learning_script import static_video_detection as deep_learning_static_detection
from detection_scripts.physiological_signal_script import run_detection
from detection_scripts.audio_analysis_script import predict_audio
from detection_scripts.visual_artifacts_script import run_visual_artifacts_detection as visual_artifacts_static_detection
from detection_scripts.legacy.body_posture_script import detect_body_posture, body_posture_live_detection
from detection_scripts.physiological_signal_ml import run_detection as run_physio_ml

app = Flask(__name__)

# Dynamically get the absolute path to the 'static/uploads/' directory
upload_folder = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder

# Create the directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def get_video_fps(video_path, default_fps=30):
    """
    Returns the framerate (as a float) of the given video file using ffprobe.
    If detection fails, returns default_fps.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-print_format', 'json',
        '-show_entries', 'stream=avg_frame_rate',
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        fps_str = info['streams'][0]['avg_frame_rate']
        num, denom = map(int, fps_str.split('/'))
        fps = num / denom if denom != 0 else 0
        if fps == 0:
            return default_fps
        return fps
    except Exception:
        return default_fps

def convert_webm_to_mp4(original_path, mp4_path):
    fps = get_video_fps(original_path)
    cmd = [
        'ffmpeg', '-y', '-i', original_path,
        '-r', str(fps),  # use detected FPS
        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        mp4_path
    ]
    subprocess.run(cmd, check=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wip')
def wip():
    return render_template('wip.html')

@app.route('/contact')
def contact():
    return render_template('wip.html')

@app.route('/navbar')
def navbar():
    return render_template('includes/navbar.html')

@app.route('/footer')
def footer():
    return render_template('includes/footer.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/creation')
def creation():
    return render_template('creation.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/education')
def education():
    return render_template('education.html')

@app.route('/challenge')
def challenge():
    return render_template('challenge.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/dragdrop')
def dragdrop():
    return render_template('dragdrop.html')

@app.route('/stages')
def stages():
    return render_template('stages.html')

@app.route('/dialogues')
def dialogues():
    stage = request.args.get('stage', default=1, type=int)
    return render_template('dialogues.html', stage=stage)

@app.route('/fight')
def fight():
    stage = request.args.get('stage', default=1, type=int)
    return render_template('fight.html', stage=stage)

@app.route('/endgame')
def endgame():
    return render_template('endgame.html')

@app.route('/history_of_deepfakes')
def history_of_deepfakes():
    return render_template('history_of_deepfakes.html')

@app.route('/impact_on_society')
def impact_on_society():
    return render_template('impact_on_society.html')

@app.route('/legal_and_ethics')
def legal_and_ethics():
    return render_template('legal_and_ethics.html')

@app.route('/future_trends')
def future_trends():
    return render_template('future_trends.html')

@app.route('/case_studies')
def case_studies():
    return render_template('case_studies.html')

@app.route('/identify_deepfakes')
def identify_deepfakes():
    return render_template('identify_deepfakes.html')

@app.route('/protect_yourself')
def protect_yourself():
    return render_template('protect_yourself.html')

@app.route('/deep_learning_based_try')
def deep_learning_based_try():
    return render_template('deep_learning_based_try.html')

@app.route('/visual_artifacts_try')
def visual_artifacts_try():
    return render_template('visual_artifacts_try.html')

@app.route('/deep_learning_based_detection', methods=['GET', 'POST'])
def deep_learning_based_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            result = deep_learning_live_detection(filename)
            return render_template('result.html', result=result, analysis_type='deep_learning')
    return render_template('deep_learning_based_detection.html')

@app.route('/physiological_signal_analysis', methods=['GET'])
def physiological_signal_analysis():
    return render_template('physiological_signal_analysis.html')

@app.route('/physiological_signal_try', methods=['GET', 'POST'])
def physiological_signal_try():
    output_folder = 'static/results'
    os.makedirs(output_folder, exist_ok=True)
    if request.method == 'POST':
        file = request.files['file']
        detection_method = request.form.get('detection_method', 'deep')
        if file:
            # Making the uploaded videos uniquely named
            video_tag = uuid.uuid4().hex
            name, ext = os.path.splitext(file.filename)
            new_file_name = video_tag + ext
            filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
            file.save(filename)

            # convert to mp4 for better compatibility
            if not filename.lower().endswith('.mp4'):
                mp4_path = os.path.splitext(filename)[0] + '.mp4'
                convert_webm_to_mp4(filename, mp4_path)
                video_path_for_processing = mp4_path
                if os.path.exists(filename) and os.path.exists(mp4_path) and filename != mp4_path:
                    os.remove(filename)
            else:
                video_path_for_processing = filename
                mp4_path = ""

            # === Call the respective detection function ===
            if detection_method == "machine":
                # Import if not already at top: from physiological_signal_ml import run_detection as run_physio_ml
                face_results, output_video = run_physio_ml(video_path_for_processing, video_tag=video_tag)
            else:
                # Default: deep learning
                face_results, output_video = run_detection(video_path_for_processing, video_tag=video_tag)

            if os.path.exists(filename):
                os.remove(filename)
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
                
            return render_template(
                'result.html',
                analysis_type='physiological',
                face_results=face_results,
                output_video=output_video,
                physio_method=detection_method
            )
    return render_template('physiological_signal_try.html')

@app.route('/start_real_time_detection', methods=['POST'])
def start_real_time_detection():
    output_folder = 'static/results'
    os.makedirs(output_folder, exist_ok=True)
    face_results, output_video  = run_detection(0, is_webcam=True)
    return render_template('result.html', analysis_type='physiological', face_results=face_results, output_video=output_video)

@app.route('/visual_artifacts_detection')
def visual_artifacts_detection():
    return render_template('visual_artifacts_detection.html')

@app.route('/real_time_detection', methods=['GET'])
def real_time_detection_page():
    return render_template('real_time_detection.html')

@app.route('/audio_analysis_page', methods=['GET'])
def audio_analysis_page():
    return render_template('audio_analysis.html')

@app.route('/audio_analysis_try_page', methods=['GET'])
def audio_analysis_try_page():
    return render_template('audio_analysis_try.html')
    
@app.route('/deep_learning_static', methods=['GET', 'POST'])
def deep_learning_static():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    # Making the uploaded videos uniquely named
    video_tag = uuid.uuid4().hex
    name, ext = os.path.splitext(file.filename)
    new_file_name = video_tag + ext
    filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
    file.save(filename)

    # convert to mp4 for better compatibility
    if not filename.lower().endswith('.mp4'):
        mp4_path = os.path.splitext(filename)[0] + '.mp4'
        convert_webm_to_mp4(filename, mp4_path)
        video_path_for_processing = mp4_path
        if os.path.exists(filename) and os.path.exists(mp4_path) and filename != mp4_path:
            os.remove(filename)
    else:
        video_path_for_processing = filename
        mp4_path = ""

    output_folder = "static/results/"
    result, real_count, fake_count, rvf_plot, conf_plot = deep_learning_static_detection(video_path_for_processing, output_folder, video_tag)

    return render_template('result.html', analysis_type='deep_learning_static', result=result, real_count=real_count, fake_count=fake_count, rvf_plot=rvf_plot, conf_plot=conf_plot)


@app.route('/visual_artifacts_static', methods=['GET', 'POST'])
def visual_artifacts_static():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Making the uploaded videos uniquely named
            video_tag = uuid.uuid4().hex
            name, ext = os.path.splitext(file.filename)
            new_file_name = video_tag + ext
            filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
            file.save(filename)

            # convert to mp4 for better compatibility
            if not filename.lower().endswith('.mp4'):
                mp4_path = os.path.splitext(filename)[0] + '.mp4'
                convert_webm_to_mp4(filename, mp4_path)
                video_path_for_processing = mp4_path
                if os.path.exists(filename) and os.path.exists(mp4_path) and filename != mp4_path:
                    os.remove(filename)
            else:
                video_path_for_processing = filename
                mp4_path = ""

            face_results, output_video = visual_artifacts_static_detection(video_path_for_processing, video_tag, output_dir='static/results')
            if os.path.exists(filename):
                os.remove(filename)
            return render_template(
                'result.html',
                analysis_type='visual_artifacts',
                face_results=face_results,
                output_video=output_video
            )
    return render_template('visual_artifacts_try.html')

@app.route('/audio_analysis', methods=['GET', 'POST'])
def audio_analysis():
    if request.method == 'POST':
        file = request.files['file']
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)
        
        if file and file.filename:
            # Get the file extension
            original_filename = secure_filename(file.filename)
            file_extension = os.path.splitext(original_filename)[1]
            
            # Generate random filename with original extension
            unique_tag = uuid.uuid4().hex
            random_filename = f"{unique_tag}{file_extension}"
            filename = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
            
            file.save(filename)
            
            prediction_class, mel_spectrogram_path, mfcc_path, delta_path, f0_path, prediction_value, uploaded_audio = predict_audio(filename, output_folder, unique_tag)
            result = "Spoof" if prediction_class == 1 else "Bonafide"
            
            return render_template('result.html', analysis_type='audio', result=result, 
                                   mel_spectrogram_path=mel_spectrogram_path, 
                                   mfcc_path=mfcc_path,
                                   delta_path=delta_path,
                                   f0_path=f0_path,
                                   prediction_value=round(prediction_value*100),
                                   uploaded_audio=uploaded_audio)
    
    return render_template('audio_analysis_try.html')
    
@app.route('/delete_files', methods=['POST'])
def delete_files():
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        for folder_name in os.listdir(upload_folder):
            folder_path = os.path.join(upload_folder, folder_name)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(folder_path)
            elif os.path.isfile(folder_path):
                os.remove(folder_path)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/body_posture_analysis', methods=['GET'])
def body_posture_analysis():
    return render_template('body_posture_analysis.html')

@app.route('/body_posture_try', methods=['GET'])
def body_posture_try():
    return render_template('body_posture_try.html')

@app.route('/body_posture_detect', methods=['GET', 'POST'])
def body_posture_detect():
    if request.method == 'POST':
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)
        file = request.files['file']
        
        if file:
            # Making the uploaded videos uniquely named
            video_tag = uuid.uuid4().hex
            name, ext = os.path.splitext(file.filename)
            new_file_name = video_tag + ext
            filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
            file.save(filename)

            # Call video processing function
            result = detect_body_posture(filename, video_tag)

            if "error" in result:
                return render_template('result.html', analysis_type='body_posture', result=result["error"])

            prediction = result["prediction"]
            confidence = round(result["confidence"] * 100)

            return render_template('result.html', analysis_type='body_posture', result=prediction, confidence=confidence)

    return render_template('body_posture_analysis.html')

    
@app.route('/multi_detect', methods=['GET'])
def multi_detect():
    return render_template('multi_detection.html')

@app.route('/multi_detection', methods=['POST'])
def multi_detection():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    video_tag = uuid.uuid4().hex
    name, ext = os.path.splitext(file.filename)
    new_file_name = video_tag + ext
    filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
    file.save(filename)

    methods = request.form.getlist("methods")
    if not methods:
        return "No detection method selected", 400

    output_folder = "static/results/"

    def run_detection_wrapper(method, filename, output_folder, method_tag):
        try:
            # Directly call detection functions, as before
            if method == "deep_learning":
                result = deep_learning_static_detection(filename, output_folder, unique_tag=method_tag, method="multi")
            elif method == "physiological":
                result = run_physio_ml(filename, video_tag=method_tag, method="multi")
            elif method == "body_posture":
                result = detect_body_posture(filename, unique_tag=method_tag)
            elif method == "visual_artifacts":
                result = visual_artifacts_static_detection(filename, video_tag=method_tag, output_dir=output_folder, method="multi")
            else:
                result = "Unknown"
            return (method, result)
        except Exception as e:
            # For debugging, print the traceback
            print(f"[ERROR] {method} crashed:\n{traceback.format_exc()}")
            return (method, f"error: {e}")

    # Use a ProcessPoolExecutor to parallelize detection methods
    results = {}
    with ProcessPoolExecutor(max_workers=len(methods)) as executor:
        # Map each method to a future
        method_tag_map = {m: f"{video_tag}_{m}" for m in methods}
        future_to_method = {
            executor.submit(run_detection_wrapper, method, filename, output_folder, method_tag_map[method]): method
            for method in methods
        }
        for future in as_completed(future_to_method):
            method = future_to_method[future]
            try:
                method_result, result = future.result(timeout=300)  # 5 minutes max per method
            except Exception as exc:
                method_result, result = method, f"error: {exc}"
            results[method_result] = result

    if os.path.exists(filename):
        os.remove(filename)

    # Process results just as before
    processed_results = {}
    for method, result in results.items():
        if isinstance(result, str):
            processed_results[method] = result
        elif isinstance(result, tuple):
            processed_results[method] = result[0]
        elif isinstance(result, dict) and "prediction" in result:
            processed_results[method] = result["prediction"]
        else:
            processed_results[method] = "Unknown"

    return render_template("result.html", analysis_type='multi_detection', results=processed_results)

if __name__ == '__main__':
    # --- Comment line below to go to development, uncomment to go to production ---
    # serve(app, host="0.0.0.0", port=5000)

    # --- Comment line below to go to production, uncomment to go to development ---
    app.run(debug=True)