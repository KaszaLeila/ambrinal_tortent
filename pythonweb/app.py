from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pickle
import librosa
import numpy as np
import pdb

app = Flask(__name__)
CORS(app)

# Betöltjük a korábban kiképzett modellt
model_path = 'mlp_classifier.model'
model = pickle.load(open(model_path, 'rb'))

# Fájl mentési hely
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# A modellhez használt jellemzők
def extract_features(file_path):
    try:
        with open(file_path, "rb") as wave_file:
            signal, rate = librosa.load(file_path, sr=16000)
        
        # Kinyerjük az MFCC jellemzőket
        mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)

        # Kinyerjük a chroma jellemzőket
        chroma = librosa.feature.chroma_stft(y=signal, sr=rate)

        # Kinyerjük a mel jellemzőket
        mel = librosa.feature.melspectrogram(y=signal, sr=rate)

        # Vegyük az átlagát minden jellemző típusnak
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        mel_mean = np.mean(mel.T, axis=0)

        # Egyesítsük a jellemzőket
        features = np.hstack([mfccs_mean, chroma_mean, mel_mean])

        return np.array(features).reshape(1, -1)
    
    except Exception as e:
        pdb.post_mortem()  # Hozzáadott sor
        print(f"Error extracting features: {str(e)}")
        return None

    
# Ne mutass favicon.ico hibaüzeneteket
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    # Ellenőrizzük, hogy a kérés tartalmazza-e a 'audio' fájlt
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']

    # Ellenőrizzük, hogy a fájl neve érvényes
    if file.filename == '':
        return jsonify({'error': 'Invalid file name'}), 400

    # Feltöltjük a hangfájlt
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Kinyerjük a jellemzőket a feltöltött hangfájlból
        features = extract_features(file_path)

        if features is None:
            return jsonify({'error': 'Error extracting features from audio file'}), 500

        # Modell érzelmek predikciója
        emotion_prediction = model.predict(features)[0]

        return jsonify({'emotion': emotion_prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        return analyze_emotion()

if __name__ == '__main__':
    app.run(debug=True)
