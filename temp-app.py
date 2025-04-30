from flask import Flask, render_template, request, redirect, url_for, jsonify
import os

# Import your detection functions for each method
from detection_scripts.physiological_signal_script import detect_physiological_signal, real_time_detection
from detection_scripts.audio_analysis_script import predict_audio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history_of_deepfakes')
def history_of_deepfakes():
    return render_template('history_of_deepfakes.html')

@app.route('/impact_on_society')
def impact_on_society():
    return render_template('impact_on_society.html')

@app.route('/legal_and_ethics')
def legal_and_ethics():
    return render_template('legal_and_ethics.html')

@app.route('/technological_advancements')
def technological_advancements():
    return render_template('technological_advancements.html')

@app.route('/case_studies')
def case_studies():
    return render_template('case_studies.html')

@app.route('/physiological_signal_analysis', methods=['GET', 'POST'])
def physiological_signal_analysis():
    if request.method == 'POST':
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_folder, exist_ok=True)
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            prediction, hr, confidence, real_count, fake_count = detect_physiological_signal(filename, output_folder)
            return render_template('result.html', uploaded_file=filename, prediction=prediction, hr=hr, confidence=confidence, real_count=real_count, fake_count=fake_count, analysis_type='physiological')
    return render_template('physiological_signal_analysis.html')

@app.route('/deep_learning_based_detection', methods=['GET', 'POST'])
def deep_learning_based_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            result = detect_deep_learning(filename)
            return render_template('result.html', result=result, analysis_type='deep_learning')
    return render_template('deep_learning_based_try.html')

app.route('/physiological_signal_try', methods=['GET', 'POST'])
def physiological_signal_try():
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    os.makedirs(output_folder, exist_ok=True)
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            prediction, hr, confidence, real_count, fake_count = detect_physiological_signal(filename, output_folder)
            if prediction is not None:
                result = 'Real' if prediction == 0 else 'Fake'
                return render_template('result.html', analysis_type='physiological', result=result)
    return render_template('physiological_signal_try.html')

@app.route('/visual_artifacts_detection', methods=['GET', 'POST'])
def visual_artifacts_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            result = detect_visual_artifacts(filename)
            return render_template('result.html', result=result, analysis_type='visual_artifacts')
    return render_template('visual_artifacts_detection.html')

@app.route('/audio_analysis', methods=['GET', 'POST'])
def audio_analysis():
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    os.makedirs(output_folder, exist_ok=True)
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            prediction_class, mel_spectrogram_path = predict_audio(filename, output_folder)
            if prediction_class is not None:
                result = 'Bonafide' if prediction_class == 1 else 'Spoof'
                return render_template('result.html', analysis_type='audio', result=result, mel_spectrogram_path=mel_spectrogram_path)
            else:
                return render_template('result.html', analysis_type='audio', result="No valid segments found.", mel_spectrogram_path=None)
    return render_template('audio_analysis_try.html')


@app.route('/real_time_detection', methods=['GET'])
def real_time_detection_page():
    return render_template('real_time_detection.html')

@app.route('/start_real_time_detection', methods=['POST'])
def start_real_time_detection():
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    os.makedirs(output_folder, exist_ok=True)
    prediction, hr = real_time_detection(output_folder)
    if prediction is not None:
        result = f'The video is predicted to be {"REAL" if prediction == 0 else "FAKE"} with an average heart rate of {hr:.2f} BPM.'
    else:
        result = 'Error: Could not detect any faces or extract signals.'
    return render_template('result.html', result=result, analysis_type='real_time')

@app.route('/delete_files', methods=['POST'])
def delete_files():
    try:
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Also delete the uploaded video file
        uploaded_file = request.json.get('uploaded_file')
        if uploaded_file:
            if os.path.isfile(uploaded_file):
                os.remove(uploaded_file)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
