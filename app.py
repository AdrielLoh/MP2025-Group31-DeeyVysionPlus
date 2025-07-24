from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import logging
import subprocess
import uuid
from werkzeug.utils import secure_filename
import json
from waitress import serve
import yt_dlp

logging.basicConfig(level=logging.DEBUG)
from detection_scripts.deep_based_learning_script import live_detection as deep_learning_live_detection
from detection_scripts.deep_based_learning_script import static_video_detection as deep_learning_static_detection
from detection_scripts.physiological_signal_script import run_detection as run_dl_detection
from detection_scripts.audio_analysis_script import predict_audio
from detection_scripts.visual_artifacts_script import run_visual_artifacts_detection as visual_artifacts_static_detection
from detection_scripts.legacy.body_posture_script import body_posture_live_detection
from detection_scripts.body_posture_script import detect_body_posture
from detection_scripts.physiological_signal_ml import run_detection as run_ml_detection

app = Flask(__name__)

# Dynamically get the absolute path to the 'static/uploads/' directory
upload_folder = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# ===== TOGGLE 'True' FOR DEMO MODE 'False' FOR NORMAL MODE =====
# Demo mode disables all detection POST endpoints to prevent DoS. Only to be used when hosted website for a large audience to try
DEMO_MODE = True

# Create the directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.wav', '.flac', '.mp3', '.ogg'}
ALLOWED_MIME_TYPES = {'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'audio/mpeg', 
                      'audio/ogg', 'audio/wav', 'audio/vnd.wav',
                      'audio/flac', 'audio/x-flac'}
MAX_VIDEO_DURATION = 30 # For URL downloads only 

# ===== HELPER FUNCTIONS =====
def allowed_file(filename, mimetype):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS and mimetype in ALLOWED_MIME_TYPES

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

def trim_video(input_path, output_path, duration=30):
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-t', str(duration),
        '-c', 'copy',
        output_path
    ]
    subprocess.run(cmd, check=True)

def download_video(video_tag, video_url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(app.config['UPLOAD_FOLDER'], f'{video_tag}.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'merge_output_format': 'mp4'
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded_filename = ydl.prepare_filename(info)
            # trim long videos to 30s
            trimmed_filename = os.path.splitext(downloaded_filename)[0] + '_trimmed.mp4'
            trim_video(downloaded_filename, trimmed_filename, duration=30)
            os.remove(downloaded_filename)
            return trimmed_filename
    except Exception as e:
        return f"Failed to download video from URL: {e}"

def detection_disabled_response():
    return render_template('detection_disabled.html'), 403

# ===== ENDPOINTS =====
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
        if DEMO_MODE:
            return detection_disabled_response()
        file = request.files.get('file')
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
        if DEMO_MODE:
            return detection_disabled_response()
        file = request.files.get('file')
        detection_method = request.form.get('detection_method', 'deep')
        video_url = request.form.get('video_url', '').strip()
        video_tag = uuid.uuid4().hex # Making the uploaded videos uniquely named

        if video_url:
            video_path_for_processing = download_video(video_tag, video_url)
            if "Failed to download" in video_path_for_processing:
                return "Invalid video URL"
            filename = video_path_for_processing
            mp4_path = ""
        elif file:
            if not allowed_file(file.filename, file.content_type):
                return "Invalid file type"
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
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
            face_results, output_video = run_ml_detection(video_path_for_processing, video_tag=video_tag)
        else:
            # Default: deep learning
            face_results, output_video = run_dl_detection(video_path_for_processing, video_tag=video_tag)

        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
            
        if request.headers.get('Accept') == 'application/json':
            # Build JSON result for extension
            real_faces = sum(1 for face in face_results if face.get('result') == 'Real')
            fake_faces = sum(1 for face in face_results if face.get('result') == 'Fake')
            overall_prediction = "Real" if real_faces > fake_faces else "Fake" if fake_faces > 0 else "Inconclusive"
            if real_faces > 0 and fake_faces > 0:
                overall_prediction = "Partial Fake"
            return jsonify({
                "prediction": overall_prediction,
                "face_results": face_results,
                "output_video": output_video
            })
        else:
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
    if DEMO_MODE:
        return detection_disabled_response()
    output_folder = 'static/results'
    os.makedirs(output_folder, exist_ok=True)
    face_results, output_video  = run_dl_detection(0, is_webcam=True)
    return render_template('result.html', analysis_type='physiological', face_results=face_results, output_video=output_video)

@app.route('/start_visual_artifacts_detection', methods=['POST'])
def start_visual_artifacts_detection():
    if DEMO_MODE:
        return detection_disabled_response()
    
    file = request.files.get('video')
    if not file or file.filename == '':
        return "No video uploaded", 400

    import uuid
    video_tag = uuid.uuid4().hex
    name, ext = os.path.splitext(file.filename)
    new_file_name = video_tag + ext
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
    file.save(upload_path)

    # Process video with visual artifact detection
    output_dir = os.path.join('static', 'results', video_tag)
    os.makedirs(output_dir, exist_ok=True)

    from detection_scripts.visual_artifacts_script import run_visual_artifacts_detection
    result = run_visual_artifacts_detection(upload_path, output_dir)

    # Optionally clean up uploaded file
    # os.remove(upload_path)
    for idx, face in enumerate(result['face_results']):
        face['display_id'] = idx + 1
        
    return render_template(
        'result.html',
        analysis_type='visual_artifacts',
        face_results=result['face_results'],
        video_with_boxes=result['video_with_boxes']
    )

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
    if request.method == "POST":
        if DEMO_MODE:
            return detection_disabled_response()
        file = request.files.get('file')
        video_url = request.form.get('video_url', '').strip()
        video_tag = uuid.uuid4().hex # Making the uploaded videos uniquely named

        if video_url:
            video_path_for_processing = download_video(video_tag, video_url)
            if "Failed to download" in video_path_for_processing:
                return "Invalid video URL"
        elif file:
            file = request.files.get('file')

            if not allowed_file(file.filename, file.content_type):
                return "Invalid file type"
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
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

        if request.headers.get('Accept') == 'application/json':
            return jsonify({
                "prediction": result,
                "real_count": real_count,
                "fake_count": fake_count,
                "rvf_plot": rvf_plot,
                "conf_plot": conf_plot
            })
        else:
            return render_template(
                'result.html', 
                analysis_type='deep_learning_static', 
                result=result, 
                real_count=real_count, 
                fake_count=fake_count, 
                rvf_plot=rvf_plot, 
                conf_plot=conf_plot
            )

@app.route('/visual_artifacts_static', methods=['GET', 'POST'])
def visual_artifacts_static():
    output_folder = 'static/results'
    os.makedirs(output_folder, exist_ok=True)
    if request.method == 'POST':
        if DEMO_MODE:
            return detection_disabled_response()
        file = request.files.get('file')
        video_url = request.form.get('video_url', '').strip()
        video_tag = uuid.uuid4().hex  # unique
        filename = ""
        mp4_path = ""

        if video_url:
            video_path_for_processing = download_video(video_tag, video_url)
            if "Failed to download" in video_path_for_processing:
                return render_template('visual_artifacts_try.html', error="Invalid video URL.")
            filename = video_path_for_processing
            mp4_path = ""
        elif file:
            if not allowed_file(file.filename, file.content_type):
                return render_template('visual_artifacts_try.html', error="Invalid file type.")
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
            new_file_name = video_tag + ext
            filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
            file.save(filename)

            # Convert to mp4 if needed
            if not filename.lower().endswith('.mp4'):
                mp4_path = os.path.splitext(filename)[0] + '.mp4'
                convert_webm_to_mp4(filename, mp4_path)
                video_path_for_processing = mp4_path
                if os.path.exists(filename) and os.path.exists(mp4_path) and filename != mp4_path:
                    os.remove(filename)
            else:
                video_path_for_processing = filename
                mp4_path = ""
        else:
            return render_template('visual_artifacts_try.html', error="Please upload a file or provide a URL.")

        # Process with your detection function
        output_dir = os.path.join('static', 'results', video_tag)
        os.makedirs(output_dir, exist_ok=True)
        result = visual_artifacts_static_detection(video_path_for_processing, video_tag, output_dir=output_dir)

        if filename and os.path.exists(filename):
            os.remove(filename)
        if mp4_path and os.path.exists(mp4_path):
            os.remove(mp4_path)

        if request.headers.get('Accept') == 'application/json':
            # Example JSON response, adjust as needed
            overall_prediction = "Fake" if any(f.get('result') == 'Fake' for f in result.get('face_results', [])) else "Real"
            return jsonify({
                "prediction": overall_prediction,
                "face_results": result.get('face_results', []),
                "video_with_boxes": result.get('video_with_boxes')
            })
        else:
            return render_template(
                'result.html',
                analysis_type='visual_artifacts',
                face_results=result.get('face_results', []),
                video_with_boxes=result.get('video_with_boxes'),
            )
    return render_template('visual_artifacts_try.html')

@app.route('/audio_analysis', methods=['GET', 'POST'])
def audio_analysis():
    if request.method == 'POST':
        if DEMO_MODE:
            return detection_disabled_response()
        
        file = request.files.get('file')
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)
        
        if file and file.filename:
            if not allowed_file(file.filename, file.content_type):
                return "Invalid file type"
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
            if request.headers.get('Accept') == 'application/json':
                return jsonify({
                    "prediction": result,
                    "mel_spectrogram_path": mel_spectrogram_path,
                    "mfcc_path": mfcc_path
                })
            else:
                return render_template(
                    'result.html', analysis_type='audio', result=result, 
                    mel_spectrogram_path=mel_spectrogram_path, 
                    mfcc_path=mfcc_path,
                    delta_path=delta_path,
                    f0_path=f0_path,
                    prediction_value=round(prediction_value*100),
                    uploaded_audio=uploaded_audio
                )
    
    return render_template('audio_analysis_try.html')
    
@app.route('/delete_files', methods=['POST'])
def delete_files():
    if DEMO_MODE:
        return detection_disabled_response()
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
        if DEMO_MODE:
            return detection_disabled_response()
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)
        file = request.files.get('file')
        video_url = request.form.get('video_url', '').strip()
        video_tag = uuid.uuid4().hex # Making the uploaded videos uniquely named
        
        if video_url:
            filename = download_video(video_tag, video_url)
            if "Failed to download" in filename:
                return "Invalid video URL"
        elif file:
            if not allowed_file(file.filename, file.content_type):
                return "Invalid file type"
            filename = secure_filename(file.filename)
            # Making the uploaded videos uniquely named
            name, ext = os.path.splitext(filename)
            new_file_name = video_tag + ext
            filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
            file.save(filename)

            # Call video processing function
            from detection_scripts.body_posture_script import PersonTracker
            results, overall_result = detect_body_posture(filename)

            if "error" in results:
                print("error")
                return render_template('result.html', analysis_type='body_posture', results=results["error"])

            return render_template('result.html', analysis_type='body_posture', results=results, overall_result=overall_result)

    return render_template('body_posture_analysis.html')

    
@app.route('/multi_detect', methods=['GET'])
def multi_detect():
    return render_template('multi_detection.html')

@app.route('/multi_detection', methods=['POST'])
def multi_detection():
    if DEMO_MODE:
        return detection_disabled_response()
    file = request.files.get("file")
    video_url = request.form.get('video_url', '').strip()
    video_tag = uuid.uuid4().hex # Making the uploaded videos uniquely named

    if video_url:
        filename = download_video(video_tag, video_url)
        if "Failed to download" in filename:
            return "Invalid video URL"
    elif file:
        if not allowed_file(file.filename, file.content_type):
            return "Invalid file type"
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        new_file_name = video_tag + ext
        filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
        file.save(filename)

    methods = request.form.getlist("methods")
    if not methods:
        return "No detection method selected", 400

    output_folder = "static/results/"

    method_tag_map = {m: f"{video_tag}_{m}" for m in methods}

    # Run detection methods one after another, not in parallel
    results = {}
    for method in methods:
        method_tag = method_tag_map[method]
        try:
            if method == "audio":
                prediction_class, mel_spectrogram_path, mfcc_path, delta_path, f0_path, prediction_value, uploaded_audio = predict_audio(
                    filename, output_folder, unique_tag=method_tag
                )
                if prediction_class is not None:
                    result = {
                        "prediction": "Fake" if prediction_class == 1 else "Real",
                        "mel_spectrogram_path": mel_spectrogram_path,
                        "mfcc_path": mfcc_path,
                        "delta_path": delta_path,
                        "f0_path": f0_path,
                        "prediction_value": round(prediction_value * 100),
                        "type": "audio"
                    }
                else:
                    result = {"prediction": "No audio detected", "type": "audio"}
            elif method == "deep_learning":
                result = deep_learning_static_detection(filename, output_folder, unique_tag=method_tag, method="multi")
            elif method == "physiological":
                result = run_ml_detection(filename, video_tag=method_tag, method="multi")
            elif method == "body_posture":
                result = detect_body_posture(filename, unique_tag=method_tag)
            elif method == "visual_artifacts":
                result = visual_artifacts_static_detection(filename, video_tag=method_tag, output_dir=output_folder, method="multi")
            else:
                result = "Unknown"
            results[method] = result
        except Exception as e:
            import traceback
            print(f"[ERROR] {method} crashed:\n{traceback.format_exc()}")
            results[method] = f"error: {e}"

    if os.path.exists(filename):
        os.remove(filename)

    # Process results to maintain full data structure for each method
    processed_results = {}
    
    for method, result in results.items():
        if isinstance(result, str):
            if result.startswith("error:"):
                processed_results[method] = {"error": result, "prediction": "Error"}
            else:
                processed_results[method] = {"prediction": result}
        
        elif method == "deep_learning":
            # Deep learning returns: (detection_result, real_count, fake_count, rvf_plot, conf_plot)
            if isinstance(result, tuple) and len(result) >= 5:
                detection_result, real_count, fake_count, rvf_plot, conf_plot = result
                processed_results[method] = {
                    "prediction": detection_result,
                    "real_count": real_count,
                    "fake_count": fake_count,
                    "rvf_plot": rvf_plot,
                    "conf_plot": conf_plot,
                    "type": "deep_learning"
                }
            else:
                processed_results[method] = {"prediction": str(result), "error": "Invalid result format"}
        
        elif method == "physiological":
            # Physiological returns: (face_results, output_video)
            if isinstance(result, tuple) and len(result) >= 2:
                face_results, output_video = result
                # Determine overall prediction from face results
                if face_results:
                    real_faces = sum(1 for face in face_results if face.get('result') == 'Real')
                    fake_faces = sum(1 for face in face_results if face.get('result') == 'Fake')
                    overall_prediction = "Real" if real_faces > fake_faces else "Fake" if fake_faces > 0 else "Inconclusive"
                else:
                    overall_prediction = "No faces detected"
                
                processed_results[method] = {
                    "prediction": overall_prediction,
                    "face_results": face_results,
                    "output_video": output_video,
                    "type": "physiological"
                }
            else:
                processed_results[method] = {"prediction": str(result), "error": "Invalid result format"}
        
        elif method == "visual_artifacts":
            # Visual artifacts returns: (face_results, output_video)
            if isinstance(result, tuple) and len(result) >= 2:
                face_results, output_video = result
                # Determine overall prediction from face results
                if face_results:
                    real_faces = sum(1 for face in face_results if face.get('result') == 'Real')
                    fake_faces = sum(1 for face in face_results if face.get('result') == 'Fake')
                    overall_prediction = "Real" if real_faces > fake_faces else "Fake" if fake_faces > 0 else "Inconclusive"
                else:
                    overall_prediction = "No faces detected"
                
                processed_results[method] = {
                    "prediction": overall_prediction,
                    "face_results": face_results,
                    "output_video": output_video,
                    "type": "visual_artifacts"
                }
            else:
                processed_results[method] = {"prediction": str(result), "error": "Invalid result format"}
        
        elif method == "body_posture":
            # Body posture returns: {"prediction": result, "confidence": confidence}
            if isinstance(result, dict) and "prediction" in result:
                processed_results[method] = {
                    "prediction": result["prediction"],
                    "confidence": result.get("confidence"),
                    "type": "body_posture"
                }
            else:
                processed_results[method] = {"prediction": str(result), "error": "Invalid result format"}

        elif method == "audio":
            # Audio result as defined above
            if isinstance(result, dict):
                processed_results[method] = result
            else:
                processed_results[method] = {"prediction": "No result", "type": "audio"}

        else:
            # Fallback for unknown methods
            processed_results[method] = {"prediction": str(result)}

    if request.headers.get('Accept') == 'application/json':
        return jsonify(processed_results)
    return render_template("result.html", analysis_type='multi_detection', multi_results=processed_results)

if __name__ == '__main__':
    # --- Comment line below to go to development, uncomment to go to production ---
    # serve(app, host="0.0.0.0", port=5000)

    # --- Comment line below to go to production, uncomment to go to development ---
    app.run(debug=True)