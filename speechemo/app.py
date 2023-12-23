from flask import Flask, render_template, request
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from utils import extract_feature
import pickle
from flask import Flask, render_template, request
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import pickle  # Új sor: Importáljuk a pickle modult
from utils import extract_feature


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Load the trained model
model_path = "result/mlp_classifier.model"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Initialize the HashingVectorizer with the same configuration
hashing_vectorizer = HashingVectorizer(n_features=180)  # You may need to adjust n_features based on your model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No selected file")

    if file and allowed_file(file.filename):
        recording_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_recording.wav')
        file.save(recording_path)

        # Features extraction using HashingVectorizer
        features = hashing_vectorizer.transform([extract_feature(recording_path, mfcc=True, chroma=True, mel=True)])

        # Model prediction
        result = model.predict(features)[0]

        return render_template('index.html', result=result)

    return render_template('index.html', error="Invalid file format")

if __name__ == '__main__':
    app.run(debug=True)
