from flask import Flask, render_template, request, jsonify, url_for
import os
import logging
import json

logging.basicConfig(level=logging.DEBUG)
from detection_scripts.deep_based_learning_script import live_detection as deep_learning_live_detection
from detection_scripts.deep_based_learning_script import static_video_detection as deep_learning_static_detection
from detection_scripts.physiological_signal_script import run_detection
from detection_scripts.audio_analysis_script import predict_audio, predict_real_time_audio
from detection_scripts.visual_artifacts_script import live_detection as visual_artifacts_live_detection
from detection_scripts.visual_artifacts_script import static_video_detection as visual_artifacts_static_detection
from detection_scripts.body_posture_script import detect_body_posture, body_posture_live_detection


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

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

@app.route('/physiological_signal_analysis', methods=['GET', 'POST'])
def physiological_signal_analysis():
    output_folder = 'static/results'
    os.makedirs(output_folder, exist_ok=True)
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            prediction, hr, confidence, real_count, fake_count, signal_plot, spectrum_plot, output_video  = run_detection(filename, is_webcam=False)
            return render_template('result.html', analysis_type='physiological', result=prediction, hr=hr, confidence=confidence,
                                   real_count=real_count, fake_count=fake_count, signal_plot=signal_plot, spectrum_plot=spectrum_plot, output_video=output_video)
    return render_template('physiological_signal_analysis.html')

@app.route('/physiological_signal_try', methods=['GET', 'POST'])
def physiological_signal_try():
    output_folder = 'static/results'
    os.makedirs(output_folder, exist_ok=True)
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            prediction, hr, confidence, real_count, fake_count, signal_plot, spectrum_plot, output_video  = run_detection(filename, is_webcam=False)
            return render_template('result.html', analysis_type='physiological', result=prediction, hr=hr, confidence=confidence,
                                   real_count=real_count, fake_count=fake_count, signal_plot=signal_plot, spectrum_plot=spectrum_plot, output_video=output_video)
    return render_template('physiological_signal_try.html')

@app.route('/start_real_time_detection', methods=['POST'])
def start_real_time_detection():
    output_folder = 'static/results'
    os.makedirs(output_folder, exist_ok=True)
    prediction, hr, confidence, real_count, fake_count, signal_plot, spectrum_plot, output_video  = run_detection(0, is_webcam=True)
    return render_template('result.html', analysis_type='physiological', result=prediction, hr=hr, confidence=confidence,
                                   real_count=real_count, fake_count=fake_count, signal_plot=signal_plot, spectrum_plot=spectrum_plot, output_video=output_video)

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

@app.route('/start_real_time_deep_learning_detection', methods=['POST'])
def start_real_time_deep_learning_detection():
    try:
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

        overall_result, real_count, fake_count = deep_learning_live_detection(output_folder)
        
        return render_template('result.html', analysis_type='deep_learning', result=overall_result, real_count=real_count, fake_count=fake_count)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/deep_learning_static', methods=['GET', 'POST'])
def deep_learning_static():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    output_folder = "static/results/"
    result, real_count, fake_count = deep_learning_static_detection(filename, output_folder)

    return render_template('result.html', analysis_type='deep_learning_static', result=result, real_count=real_count, fake_count=fake_count)


@app.route('/start_visual_artifacts_detection', methods=['POST'])
def start_visual_artifacts_detection():
    try:
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

        overall_result, real_count, fake_count = visual_artifacts_live_detection(output_folder)
        
        if overall_result is None:
            raise ValueError("Live detection failed to produce a result.")

        return render_template('result.html', analysis_type='visual_artifacts', result=overall_result, real_count=real_count, fake_count=fake_count)
    except Exception as e:
        logging.error(f"Error during visual artifacts detection: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/visual_artifacts_static', methods=['GET', 'POST'])
def visual_artifacts_static():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    output_folder = "static/results/"
    overall_results, real_frame_count, fake_frame_count = visual_artifacts_static_detection(filename, output_folder)

    return render_template('result.html', analysis_type='visual_artifact_static', result=overall_results, real_count=real_frame_count, fake_count=fake_frame_count)


@app.route('/audio_analysis', methods=['GET', 'POST'])
def audio_analysis():
    if request.method == 'POST':
        file = request.files['file']
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            logging.debug(f"File saved: {filename}")
            
            prediction_class, mel_spectrogram_path, mfcc_path, delta_path, f0_path, prediction_value = predict_audio(filename, output_folder)
            result = "Spoof" if prediction_class == 1 else "Bonafide"
            logging.debug(f"Prediction: {result}, Mel Spectrogram Path: {mel_spectrogram_path}")
            
            return render_template('result.html', analysis_type='audio', result=result, 
                                   mel_spectrogram_path=mel_spectrogram_path, 
                                   mfcc_path=mfcc_path,
                                   delta_path=delta_path,
                                   f0_path=f0_path,
                                   prediction_value=prediction_value)
    return render_template('audio_analysis_try.html')

@app.route('/start_real_time_audio_analysis', methods=['POST'])
def start_real_time_audio_analysis():
    try:
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)
        
        prediction_class, mel_spectrogram_path, mfcc_path, delta_path, f0_path, prediction_value = predict_real_time_audio(output_folder)
        result = "Spoof" if prediction_class == 1 else "Bonafide"
        logging.debug(f"Real-time Prediction: {result}, Mel Spectrogram Path: {mel_spectrogram_path}")
        
        if prediction_class is not None:
            return render_template('result.html', analysis_type='audio', result=result, 
                                   mel_spectrogram_path=mel_spectrogram_path, 
                                   mfcc_path=mfcc_path,
                                   delta_path=delta_path,
                                   f0_path=f0_path,
                                   prediction_value=prediction_value)
        else:
            return render_template('result.html', analysis_type='audio', result="Error in detection", mel_spectrogram_path=None, 
                                   mfcc_path=None,
                                   delta_path=None,
                                   f0_path=None,
                                   prediction_value=0)
    except Exception as e:
        logging.error(f"Error during real-time audio analysis: {e}")
        return render_template('result.html', analysis_type='audio', result="Error in detection", mel_spectrogram_path=None, 
                                   mfcc_path=None,
                                   delta_path=None,
                                   f0_path=None,
                                   prediction_value=0)
    
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
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Call video processing function
            result = detect_body_posture(filename)

            if "error" in result:
                return render_template('result.html', analysis_type='body_posture', result=result["error"])

            prediction = result["prediction"]
            confidence = result["confidence"]
            

            if prediction == "Real":
                result_message = f'The video is predicted to be REAL.'
            elif prediction == "Fake":
                result_message = f'The video is predicted to be FAKE.'

            return render_template('result.html', analysis_type='body_posture', result=result_message, confidence=confidence)

    return render_template('body_posture_analysis.html')


@app.route('/start_real_time_body_posture_detection', methods=['POST'])
def start_real_time_body_posture_detection():
    try:
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

        overall_result, real_count, fake_count = body_posture_live_detection(output_folder)
        
        return render_template('result.html', analysis_type='body_posture_live', result=overall_result, real_count=real_count, fake_count=fake_count)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/multi_detect', methods=['GET'])
def multi_detect():
    return render_template('multi_detection.html')


@app.route('/multi_detection', methods=['POST'])
def multi_detection():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filename)

    
    methods = request.form.getlist("methods")  # Use getlist()
    print("Raw methods data:", methods)  # Debugging

    if not methods:
        return "No detection method selected", 400

    output_folder = "static/results/"
    raw_results = {}

    for method in methods:
        print(f"Processing: {method}...")  # Debugging output

        if method == "deep_learning":
            raw_results["deep_learning"] = deep_learning_static_detection(filename, output_folder)

        elif method == "physiological":
            raw_results["physiological"] = detect_physiological_signal(filename, output_folder)

        elif method == "body_posture":
            raw_results["body_posture"] = detect_body_posture(filename)

        elif method == "visual_artifacts":
            raw_results["visual_artifacts"] = visual_artifacts_static_detection(filename, output_folder)

    print("Final Results:", raw_results)  # Debugging output

    # Extract only "Real" or "Fake" values before passing to template
    processed_results = {}
    for method, result in raw_results.items():
        if isinstance(result, str):  # If already "Real" or "Fake"
            processed_results[method] = result
        elif isinstance(result, tuple):  # If tuple, take first value
            processed_results[method] = result[0]
        elif isinstance(result, dict) and "prediction" in result:  # If dictionary, extract prediction key
            processed_results[method] = result["prediction"]
        else:
            processed_results[method] = "Unknown"  # Fallback case

    print("Processed Results:", processed_results)  # Debugging output

    return render_template("result.html", analysis_type='multi_detection', results=processed_results)

if __name__ == '__main__':
    app.run(debug=True)
