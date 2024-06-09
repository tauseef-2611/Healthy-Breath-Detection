from flask import Flask, request, send_file
from flask_cors import CORS
import os
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
from ViT import Pred
from flask import jsonify


app = Flask(__name__)
CORS(app) 


import traceback

import os
import uuid
import matplotlib
matplotlib.use('Agg')

model=Pred("model/model.keras")

# imageName="error.png"
def save_spectrogram(audio_path, output_path, sr=4000):
    try:
        print(f"Processing {audio_path}")
        print(f"Output path: {output_path}")
        
        x, sr = librosa.load(audio_path, sr=sr)
        X = librosa.stft(x)  # Apply Short-Time Fourier Transform (STFT)
        Xdb = librosa.amplitude_to_db(abs(X))
        
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr)
        plt.axis('off')  # Turn off the axis

        # if output_path is None:
        #     # Generate a unique filename with a .png extension
        #     output_path = f"{uuid.uuid4().hex}.png"

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Saved spectrogram to {output_path} in directory {os.getcwd()}")
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        traceback.print_exc()  # Print the full traceback

@app.route('/', methods=['GET'])
def home():
    return "Hello, World!"

@app.route('/echo', methods=['POST'])
def echo():
    data = request.get_json()
    return data


@app.route('/upload_image', methods=['POST'])
def upload_image():
    image = request.files['image']
    image.save(os.path.join('images', image.filename))
    time.sleep(5)
    return "Hello"

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    global imageName
    if 'audio' in request.files:
        audio = request.files['audio']
        audio.save(audio.filename)
        try:
            imageName=os.path.splitext(audio.filename)[0] + ".png"
            save_spectrogram(audio.filename,  imageName)
            return "Ok", 200
        except Exception as e:
            return str(e), 500
    

@app.route('/upload_blob', methods=['POST'])
def upload_blob():
    audio_data = request.get_data()
    with open('the.wav', 'wb') as audio_file:
        audio_file.write(audio_data)
    return "Audio file saved", 200




@app.route('/predict',methods=['GET'])
def predict():
    global imageName

    classp, predval = model.predict(imageName)
    # classp, predval = "healthy", 0.95
    print("ImageName in predict func",imageName)

    if classp == 'Healthy':
        return jsonify({"status": "Healthy", "probability": str(predval[0])})
    
    return jsonify({"status": "Unhealthy", "probability": str(predval[0])})



@app.route("/get_image")
def get_spectorgram():
    image_path=os.path.join('output.png')
    return send_file(image_path)


if __name__ == '__main__':
    app.run(debug=True)