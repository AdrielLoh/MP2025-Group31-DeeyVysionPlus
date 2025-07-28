from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import logging
import subprocess
import uuid
from werkzeug.utils import secure_filename
import json
from waitress import serve
import yt_dlp
import hashlib
import datetime
from collections import defaultdict
from flask import send_file
import uuid

logging.basicConfig(level=logging.DEBUG)
from detection_scripts.deep_based_learning_script import static_video_detection as deep_learning_static_detection
from detection_scripts.physiological_signal_script import run_detection as run_dl_detection
from detection_scripts.audio_analysis_script import predict_audio
from detection_scripts.visual_artifacts_script import run_visual_artifacts_detection as visual_artifacts_static_detection
from detection_scripts.body_posture_script import detect_body_posture
from detection_scripts.physiological_signal_ml import run_detection as run_ml_detection

app = Flask(__name__)

# Dynamically get the absolute path to the 'static/uploads/' directory
upload_folder = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
DETECTION_LOG_PATH = os.path.join('static', 'results', 'detection_log.jsonl')

# ===== TOGGLE 'True' FOR DEMO MODE 'False' FOR NORMAL MODE =====
# Demo mode disables all detection POST endpoints to prevent DoS. Only to be used when hosted website for a large audience to try
DEMO_MODE = False

# Create the directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.wav', '.flac', '.mp3', '.ogg'}
ALLOWED_MIME_TYPES = {'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'audio/mpeg', 
                      'audio/ogg', 'audio/wav', 'audio/vnd.wav',
                      'audio/flac', 'audio/x-flac'}
MAX_VIDEO_DURATION = 30 # For URL downloads only
VALID_VIDEOS = ['.mp4', '.mov', '.avi', '.webm']

# ===== HELPER FUNCTIONS =====
def log_detection(filename, sha256, output_folders, uploaded_path, multi_predictions=None):
    entry = {
        "uuid": str(uuid.uuid4()),
        "datetime": datetime.datetime.now().astimezone().isoformat(),
        "filename": filename,
        "uploaded_path": uploaded_path,
        "sha256": sha256
    }
    # output_folders: either a string (single detection) or dict (multi)
    if isinstance(output_folders, dict):
        entry["output_folders"] = output_folders
        entry["multi_predictions"] = multi_predictions
    else:
        entry["output_folder"] = output_folders
    with open(DETECTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

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

def sha256_of_file(filepath, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_single_detection_results(history_entry):
    """Load results for single detection method"""
    output_folder = history_entry.get('output_folder')
    cached_results_path = os.path.join(output_folder, 'cached_results.json')
    
    if not os.path.exists(cached_results_path):
        return "Cached results not found", 404
    
    with open(cached_results_path, 'r') as f:
        cached_data = json.load(f)
    
    # Determine detection type from folder path
    if 'deep_learning' in output_folder:
        return render_template(
            'result.html',
            analysis_type='deep_learning_static',
            face_results=cached_data.get('face_results', []),
            output_video=cached_data.get('output_path')
        )
    elif 'audio' in output_folder:
        prediction_class = cached_data.get('prediction_class')
        if prediction_class == 1:
            result = 'Spoof'
        elif prediction_class == 0:
            result = 'Bonafide'
        else:
            result = 'Unknown'
        return render_template(
            'result.html',
            analysis_type='audio',
            result=result,
            mel_spectrogram_path=cached_data.get('mel_spectrogram_path'),
            mfcc_path=cached_data.get('mfcc_path'),
            delta_path=cached_data.get('delta_path'),
            f0_path=cached_data.get('f0_path'),
            prediction_value=round(cached_data.get('prediction_value', 0) * 100),
            uploaded_audio=cached_data.get('uploaded_audio')
        )
    elif 'visual_artifacts' in output_folder:
        return render_template(
            'result.html',
            analysis_type='visual_artifacts',
            face_results=cached_data.get('face_results', []),
            video_with_boxes=cached_data.get('video_with_boxes')
        )
    elif 'body_posture' in output_folder:
        return render_template(
            'result.html',
            analysis_type='body_posture',
            results=cached_data.get('results', {}),
            overall_result=cached_data.get('overall_result', {})
        )
    elif 'physio' in output_folder:
        return render_template(
            'result.html',
            analysis_type='physiological',
            face_results=cached_data.get('face_results', []),
            output_video=cached_data.get('output_path'),
            physio_method='machine' if 'physio_ml' in output_folder else 'deep'
        )
    else:
        return "Unknown detection type", 400

def load_multi_detection_results(history_entry):
    """Load results for multi-detection using both cached files and logged predictions"""
    output_folders = history_entry.get('output_folders', {})
    multi_predictions = history_entry.get('multi_predictions', {})
    multi_results = {}
    
    for method, folder_path in output_folders.items():
        cached_results_path = os.path.join(folder_path, 'cached_results.json')
        
        if os.path.exists(cached_results_path):
            with open(cached_results_path, 'r') as f:
                cached_data = json.load(f)
            
            # Process each method's results according to its type
            if method == 'deep_learning':
                overall_prediction = multi_predictions[method]
                multi_results[method] = {
                    'prediction': overall_prediction,
                    'face_results': cached_data.get('face_results', []),
                    'output_video': cached_data.get('output_path'),
                    'type': 'deep_learning'
                }
                
            elif method == 'audio':
                prediction_class = cached_data.get('prediction_class')
                if prediction_class == 1:
                    result = 'Fake'
                elif prediction_class == 0:
                    result = 'Real'
                else:
                    result = 'Unknown'
                    
                multi_results[method] = {
                    'prediction': result,
                    'mel_spectrogram_path': cached_data.get('mel_spectrogram_path'),
                    'mfcc_path': cached_data.get('mfcc_path'),
                    'delta_path': cached_data.get('delta_path'),
                    'f0_path': cached_data.get('f0_path'),
                    'prediction_value': round(cached_data.get('prediction_value', 0) * 100),
                    'uploaded_audio': cached_data.get('uploaded_audio'),
                    'type': 'audio'
                }
                
            elif method == 'visual_artifacts':
                overall_prediction = multi_predictions[method]
                multi_results[method] = {
                    'prediction': overall_prediction,
                    'face_results': cached_data.get('face_results', []),
                    'output_video': os.path.join('static', cached_data.get('video_with_boxes')),
                    'type': 'visual_artifacts'
                }
                
            elif method == 'body_posture':
                multi_results[method] = {
                    'results': cached_data.get('results', {}),
                    'overall_result': cached_data.get('overall_result', {}),
                    'type': 'body_posture'
                }
                
            elif method == 'physiological':
                overall_prediction = multi_predictions[method]
                multi_results[method] = {
                    'prediction': overall_prediction,
                    'face_results': cached_data.get('face_results', []),
                    'output_video': cached_data.get('output_path'),
                    'type': 'physiological'
                }
        else:
            # If cached results don't exist, create error entry
            multi_results[method] = {
                'error': 'Cached results not found',
                'prediction': 'Error',
                'type': method
            }
    
    return render_template(
        'result.html',
        analysis_type='multi_detection',
        multi_results=multi_results
    )

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

@app.route('/deep_learning_based_detection')
def deep_learning_based_detection():
    return render_template('deep_learning_based_detection.html')

@app.route('/physiological_signal_analysis', methods=['GET'])
def physiological_signal_analysis():
    return render_template('physiological_signal_analysis.html')

@app.route('/detection-history')
def detection_history():
    return render_template('detection_history.html')

@app.route('/<detection_uuid>')
def view_cached_result(detection_uuid):
    """Load and display cached results for a detection history entry"""
    try:
        # Find the detection entry by UUID
        history_entry = None
        try:
            with open(DETECTION_LOG_PATH, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        if entry.get('uuid') == detection_uuid:
                            history_entry = entry
                            break
        except FileNotFoundError:
            return "Detection history not found", 404
        
        if not history_entry:
            return "Detection result not found", 404
        
        # Check if this is multi-detection or single detection
        if 'output_folders' in history_entry:
            # Multi-detection case
            return load_multi_detection_results(history_entry)
        else:
            # Single detection case
            return load_single_detection_results(history_entry)
            
    except Exception as e:
        return f"Error loading results: {str(e)}", 500

@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve media files from uploads directory"""
    try:
        # Security check - ensure the file path is within uploads
        safe_path = os.path.abspath(filename)
            
        if os.path.exists(safe_path):
            return send_file(safe_path)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error serving file: {str(e)}", 500
    
@app.route('/api/detection-history')
def api_detection_history():   
    history = []
    
    try:
        with open(DETECTION_LOG_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    history.append(entry)
    except FileNotFoundError:
        return jsonify([])
    
    # Process entries to determine detection types
    processed_entries = []
    for entry in history:
        detection_types = []
        
        if 'output_folders' in entry:
            # Multi-detection
            detection_type = 'Multi Model'
            for folder in entry['output_folders']:
                if 'body_posture' in folder:
                    detection_types.append('Body Posture')
                elif 'physiological' in folder:
                    detection_types.append('Physiological')
                elif 'audio' in folder:
                    detection_types.append('Audio')
                elif 'deep_learning' in folder:
                    detection_types.append('Deep Learning')
                elif 'visual_artifacts' in folder:
                    detection_types.append('Visual Artifacts')
                else:
                    detection_types.append('Unknown')
        else:
            # Single detection
            folder = entry.get('output_folder', '')
            if 'body_posture' in folder:
                detection_type = 'Body Posture'
            elif 'physio_deep' in folder:
                detection_type = 'Deep Physiological (Beta)'
            elif 'physio_ml' in folder:
                detection_type = 'Physiological'
            elif 'audio' in folder:
                detection_type = 'Audio'
            elif 'deep_learning' in folder:
                detection_type = 'Deep Learning'
            elif 'visual_artifacts' in folder:
                detection_type = 'Visual Artifacts'
            else:
                detection_type = 'Unknown'
        
        # Determine media type from uploaded_path
        uploaded_path = entry.get('uploaded_path', '')
        media_type = 'unknown'
        if uploaded_path:
            ext = os.path.splitext(uploaded_path)[1].lower()
            if ext in ['.mp4', '.mov', '.avi', '.webm']:
                media_type = 'video'
            elif ext in ['.wav', '.mp3', '.flac', '.ogg']:
                media_type = 'audio'
        
        processed_entry = {
            'uuid': entry.get('uuid'),
            'timestamp': entry.get('datetime'),
            'filename': entry.get('filename'),
            'detection_type': detection_type if 'output_folders' not in entry else 'Multi Model',
            'detection_methods': detection_types if detection_types else [detection_type],
            'sha256': entry.get('sha256'),
            'uploaded_path': uploaded_path,
            'media_type': media_type,
            'raw_entry': entry
        }
        processed_entries.append(processed_entry)

    # Sort by timestamp (most recent first)
    final_entries = sorted(processed_entries, key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(final_entries)

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
        filename_to_log = ""

        if video_url:
            video_path_for_processing = download_video(video_tag, video_url)
            if "Failed to download" in video_path_for_processing:
                return "Invalid video URL"
            mp4_path = ""
            filename_to_log = video_url
        elif file:
            if not allowed_file(file.filename, file.content_type):
                return "Invalid file type"
            filename = secure_filename(file.filename)
            filename_to_log = os.path.basename(filename)
            name, ext = os.path.splitext(filename)
            if ext.lower() not in VALID_VIDEOS:
                return "Content must be a video"
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

        video_hash = sha256_of_file(video_path_for_processing)
        path_to_log = video_path_for_processing

        if detection_method == "machine":
            output_dir = os.path.join('static', 'results', 'physio_ml', video_hash)
            os.makedirs(output_dir, exist_ok=True)
            face_results, output_video = run_ml_detection(video_path_for_processing, video_tag=video_hash, output_dir=output_dir)
        else:
            output_dir = os.path.join('static', 'results', 'physio_deep', video_hash)
            os.makedirs(output_dir, exist_ok=True)
            face_results, output_video = run_dl_detection(video_path_for_processing, video_tag=video_hash, output_dir=output_dir)          

        log_detection(
            filename=filename_to_log,
            sha256=video_hash,
            output_folders=output_dir,
            uploaded_path=video_path_for_processing
        )

        if request.headers.get('Accept') == 'application/json':
            real_faces = sum(1 for face in face_results if face.get('result') == 'Real')
            fake_faces = sum(1 for face in face_results if face.get('result') == 'Fake')
            overall_prediction = "Real" if real_faces > fake_faces else "Fake" if fake_faces > 0 else "Inconclusive"
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

@app.route('/visual_artifacts_detection')
def visual_artifacts_detection():
    return render_template('visual_artifacts_detection.html')

@app.route('/real_time_detection', methods=['GET'])
def real_time_detection_page():
    return render_template('real_time_detection.html')

@app.route('/audio_analysis_page', methods=['GET'])
def audio_analysis_page():
    return render_template('audio_analysis.html')
    
@app.route('/deep_learning_static', methods=['GET', 'POST'])
def deep_learning_static():
    if request.method == "POST":
        if DEMO_MODE:
            return detection_disabled_response()
        mp4_path = ""
        file = request.files.get('file')
        video_url = request.form.get('video_url', '').strip()
        video_tag = uuid.uuid4().hex # Making the uploaded videos uniquely named
        filename_to_log = ""

        if video_url:
            video_path_for_processing = download_video(video_tag, video_url)
            if "Failed to download" in video_path_for_processing:
                return "Invalid video URL"
            filename_to_log = video_url
        elif file:
            if not allowed_file(file.filename, file.content_type):
                return "Invalid file type"
            filename = secure_filename(file.filename)
            filename_to_log = os.path.basename(filename)
            name, ext = os.path.splitext(filename)
            if ext.lower() not in VALID_VIDEOS:
                return "Content must be a video"
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
        
        video_hash = sha256_of_file(video_path_for_processing)

        output_folder = os.path.join('static', 'results', 'deep_learning', video_hash)
        os.makedirs(output_folder, exist_ok=True)
        face_results, output_video = deep_learning_static_detection(video_path_for_processing, output_folder, video_hash)

        log_detection(
            filename=filename_to_log,
            sha256=video_hash,
            output_folders=output_folder,
            uploaded_path=video_path_for_processing
        )

        if request.headers.get('Accept') == 'application/json':
            # Calculate overall prediction from face results
            real_faces = sum(1 for face in face_results if face.get('result') == 'Real')
            fake_faces = sum(1 for face in face_results if face.get('result') == 'Fake')
            overall_prediction = "Real" if real_faces > fake_faces else "Fake" if fake_faces > 0 else "Inconclusive"
            
            return jsonify({
                "prediction": overall_prediction,
                "face_results": face_results,
                "output_video": output_video
            })
        else:
            return render_template(
                'result.html', 
                analysis_type='deep_learning_static', 
                face_results=face_results,
                output_video=output_video
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
        filename_to_log = ""

        if video_url:
            video_path_for_processing = download_video(video_tag, video_url)
            if "Failed to download" in video_path_for_processing:
                return render_template('visual_artifacts_try.html', error="Invalid video URL.")
            filename = video_path_for_processing
            mp4_path = ""
            filename_to_log = video_url
        elif file:
            if not allowed_file(file.filename, file.content_type):
                return render_template('visual_artifacts_try.html', error="Invalid file type.")
            filename = secure_filename(file.filename)
            filename_to_log = os.path.basename(filename)
            name, ext = os.path.splitext(filename)
            if ext.lower() not in VALID_VIDEOS:
                return "Content must be a video"
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

        video_hash = sha256_of_file(video_path_for_processing)
        # Process with your detection function
        output_dir = os.path.join('static', 'results', 'visual_artifacts', video_hash)
        os.makedirs(output_dir, exist_ok=True)
        result = visual_artifacts_static_detection(video_path_for_processing, video_hash, output_dir=output_dir)

        log_detection(
            filename=filename_to_log,
            sha256=video_hash,
            output_folders=output_dir,
            uploaded_path=video_path_for_processing
        )

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
        video_url = request.form.get('video_url', '').strip()
        unique_tag = uuid.uuid4().hex
        filename = ""
        filename_to_log = ""
        
        if video_url:
            video_path_for_processing = download_video(unique_tag, video_url)
            if "Failed to download" in video_path_for_processing:
                return "Error getting video. URL may be invalid."
            filename = video_path_for_processing
            filename_to_log = video_url
        elif file:
            if not allowed_file(file.filename, file.content_type):
                return "Invalid file type"
            # Get the file extension
            original_filename = secure_filename(file.filename)
            filename_to_log = os.path.basename(original_filename)
            file_extension = os.path.splitext(original_filename)[1]
            
            # Generate random filename with original extension
            random_filename = f"{unique_tag}{file_extension}"
            filename = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
            
            file.save(filename)

        video_hash = sha256_of_file(filename)
        output_folder = os.path.join('static', 'results', 'audio', video_hash)
        os.makedirs(output_folder, exist_ok=True)

        prediction_class, mel_spectrogram_path, mfcc_path, delta_path, f0_path, prediction_value, uploaded_audio = predict_audio(filename, output_folder, video_hash)
        result = "Spoof" if prediction_class == 1 else "Bonafide"

        log_detection(
            filename=filename_to_log,
            sha256=video_hash,
            output_folders=output_folder,
            uploaded_path=uploaded_audio
        )

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
        
        file = request.files.get('file')
        video_url = request.form.get('video_url', '').strip()
        video_tag = uuid.uuid4().hex # Making the uploaded videos uniquely named
        filename_to_log = ""

        if video_url:
            filename = download_video(video_tag, video_url)
            if "Failed to download" in filename:
                return "Invalid video URL"
            filename_to_log = video_url
        elif file:
            if not allowed_file(file.filename, file.content_type):
                return "Invalid file type"
            filename = secure_filename(file.filename)
            filename_to_log = os.path.basename(filename)
            # Making the uploaded videos uniquely named
            name, ext = os.path.splitext(filename)
            if ext.lower() not in VALID_VIDEOS:
                return "Content must be a video"
            new_file_name = video_tag + ext
            filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
            file.save(filename)
            video_path_for_processing = filename
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

        video_hash = sha256_of_file(video_path_for_processing)

        output_folder = os.path.join('static', 'results', 'body_posture', video_hash)
        os.makedirs(output_folder, exist_ok=True)

        # Call video processing function
        results, overall_result = detect_body_posture(video_path_for_processing, output_folder)

        if request.headers.get('Accept') == 'application/json':
            return jsonify({
                "person_count": overall_result.get("person_count"),
                "overall_instability": overall_result.get("overall_instability"),
                "overall_figure": overall_result.get("overall_figure"),
                "persons": results,
                "prediction" : overall_result.get("prediction"),
                "type": "body_posture"
            })
        elif "error" in results:
            return render_template('result.html', analysis_type='body_posture', results=results["error"])
        else:
            log_detection(
            filename=filename_to_log,
            sha256=video_hash,
            output_folders=output_folder,
            uploaded_path=video_path_for_processing
        )

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
    filename_to_log = ""

    if video_url:
        filename = download_video(video_tag, video_url)
        if "Failed to download" in filename:
            return "Invalid video URL"
        filename_to_log = video_url
    elif file:
        if not allowed_file(file.filename, file.content_type):
            return "Invalid file type"
        filename = secure_filename(file.filename)
        filename_to_log = os.path.basename(filename)
        name, ext = os.path.splitext(filename)
        if ext.lower() not in VALID_VIDEOS:
            return "Content must be a video"
        new_file_name = video_tag + ext
        filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
        file.save(filename)

        # Convert to mp4 if needed
        if not filename.lower().endswith('.mp4'):
            mp4_path = os.path.splitext(filename)[0] + '.mp4'
            convert_webm_to_mp4(filename, mp4_path)
            if os.path.exists(filename) and os.path.exists(mp4_path) and filename != mp4_path:
                os.remove(filename)
            filename = mp4_path
    
    video_hash = sha256_of_file(filename)

    methods = request.form.getlist("methods")
    if not methods:
        return "No detection method selected", 400

    # Run detection methods one after another, not in parallel
    results = {}
    for method in methods:
        try:
            if method == "audio":
                output_folder = os.path.join('static', 'results', 'audio', video_hash)
                prediction_class, mel_spectrogram_path, mfcc_path, delta_path, f0_path, prediction_value, uploaded_audio = predict_audio(
                    filename, output_folder, unique_tag=video_tag
                )
                if prediction_class is not None:
                    result = {
                        "prediction": "Fake" if prediction_class == 1 else "Real",
                        "mel_spectrogram_path": mel_spectrogram_path,
                        "mfcc_path": mfcc_path,
                        "delta_path": delta_path,
                        "f0_path": f0_path,
                        "prediction_value": round(prediction_value * 100),
                        "uploaded_audio": uploaded_audio,
                        "type": "audio"
                    }
                else:
                    result = {"prediction": "No audio detected", "type": "audio"}
            elif method == "deep_learning":
                output_folder = os.path.join('static', 'results', 'deep_learning', video_hash)
                result = deep_learning_static_detection(filename, output_folder, unique_tag=video_tag)
            elif method == "physiological":
                output_folder = os.path.join('static', 'results', 'physio_ml', video_hash)
                result = run_ml_detection(filename, video_tag=video_tag, output_dir=output_folder)
            elif method == "body_posture":
                output_folder = os.path.join('static', 'results', 'body_posture', video_hash)
                result = detect_body_posture(filename, output_folder)
            elif method == "visual_artifacts":
                output_folder = os.path.join('static', 'results', 'visual_artifacts', video_hash)
                result = visual_artifacts_static_detection(filename, video_tag=video_tag, output_dir=output_folder)
            else:
                result = "Unknown"
            results[method] = result
        except Exception as e:
            import traceback
            print(f"[ERROR] {method} crashed:\n{traceback.format_exc()}")
            results[method] = f"error: {e}"

    # Process results to maintain full data structure for each method
    processed_results = {}
    
    for method, result in results.items():
        if isinstance(result, str):
            if result.startswith("error:"):
                processed_results[method] = {"error": result, "prediction": "Error"}
            else:
                processed_results[method] = {"prediction": result}
        
        elif method == "deep_learning":
            # Deep learning returns: (face_results, output_video)
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
            if isinstance(result, dict):
                face_results = result.get('face_results', [])
                output_video = result.get('video_with_boxes')
            elif isinstance(result, tuple) and len(result) >= 2:
                face_results = result['face_results']
                output_video = result['video_with_boxes']
            else:
                processed_results[method] = {"prediction": str(result), "error": "Invalid result format"}
                continue
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
                "output_video": os.path.join('static', output_video),
                "type": "visual_artifacts"
            }
        
        elif method == "body_posture":
            # Body posture returns: (results_list, overall_result)
            if isinstance(result, tuple) and len(result) == 2:
                results_list, overall_result = result
                processed_results[method] = {
                    "person_count": overall_result.get("person_count"),
                    "overall_instability": overall_result.get("overall_instability"),
                    "overall_figure": overall_result.get("overall_figure"),
                    "persons": results_list,
                    "prediction" : overall_result.get("prediction"),
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

    multi_output_folders = {}
    if "audio" in methods:
        multi_output_folders["audio"] = os.path.join('static', 'results', 'audio', video_hash)
    if "deep_learning" in methods:
        multi_output_folders["deep_learning"] = os.path.join('static', 'results', 'deep_learning', video_hash)
    if "physiological" in methods:
        multi_output_folders["physiological"] = os.path.join('static', 'results', 'physio_ml', video_hash)
    if "body_posture" in methods:
        multi_output_folders["body_posture"] = os.path.join('static', 'results', 'body_posture', video_hash)
    if "visual_artifacts" in methods:
        multi_output_folders["visual_artifacts"] = os.path.join('static', 'results', 'visual_artifacts', video_hash)

    multi_predictions = {}
    for method, results in processed_results.items():
        if method in ["deep_learning", "visual_artifacts", "physiological"]:
            prediction = results.get('prediction')
            multi_predictions[method] = prediction

    log_detection(
        filename=filename_to_log,
        sha256=video_hash,
        output_folders=multi_output_folders,
        uploaded_path=filename,
        multi_predictions=multi_predictions
    )

    if request.headers.get('Accept') == 'application/json':
        return jsonify(processed_results)
    return render_template("result.html", analysis_type='multi_detection', multi_results=processed_results)

if __name__ == '__main__':
    # --- Comment line below to go to development, uncomment to go to production ---
    # serve(app, host="0.0.0.0", port=5000)

    # --- Comment line below to go to production, uncomment to go to development ---
    app.run(debug=True)