## Kelas Nama Nim
## 6C Abu Bakar Sidik 19090111
## 6C Dwi Ayu Wardani 19090042
import os
import sys
import numpy as np
from util import base64_to_pil
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file


app = Flask(__name__)

# OPTION 1: LOAD YOUR OWN TRAINED MODEL FROM URL
#model_path = get_file('final_model.h5', 'https://srv-store4.gofile.io/download/Eqn1Kr/final_model.h5')  # Model 1 : 202.5 MB (Accuracy 88%)
#model_path = get_file('model_sikat.h5', 'https://srv-store1.gofile.io/download/8ZxQiY/model_sikat.h5')  # Model 2 : 9.4MB (Accuracy 78%)
#model = load_model(model_path)

# OPTION 2: LOAD YOUR OWN TRAINED MODEL FROM LOCAL FOLDER
model = load_model('models/top-2-model-cnn.h5') # ⚠️ SESUAIKAN ⚠️

def model_predict(img, model):
    img = img.resize((224, 224))            # ⚠️ SESUAIKAN ⚠️
    
    x = image.img_to_array(img)
    x = x.reshape(-1, 224, 224, 3)
    x = x.astype('float32')
    x = x / 255.0
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        #==================================================================================#

        target_names = ['Cattleya', 'Dendrobium', 'Oncidium', 'Phalaenopsis', 'Vanda']     # ⚠️ SESUAIKAN ⚠️

        hasil_label = target_names[np.argmax(preds)]
        hasil_prob = "{:.2f}".format(100 * np.max(preds)) # 2f adalah presisi angka dibelakang koma (coba ganti jadi 0f, 3f, dst)

        #==================================================================================#

        return jsonify(result=hasil_label, probability=hasil_prob)

    return None

if __name__ == '__main__':
    # OPTION 1: NORMAL SERVE THE APP
    app.run(debug=True)
    #app.run(port=5002, threaded=False)

    # OPTION 2: SERVE THE APP WITH GEVENT
    # Setiap merubah file di main.js, ubah juga 5000, menjadi 5001, dst (alasan: cache)
    #http_server = WSGIServer(('0.0.0.0', 5050), app)
    #http_server.serve_forever()