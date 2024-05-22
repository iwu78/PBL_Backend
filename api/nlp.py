from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import sqlite3
import base64
from PIL import Image
from io import BytesIO
import urllib.request
import tensorflow as tf
import numpy as np
import pickle

caption_api = Blueprint('caption_api', __name__, url_prefix='/api/caption')
api = Api(caption_api)

class ImageCaptioningModel:
    def __init__(self, fe_model, caption_model_path, tokenizer_path, max_length_path):
        self.fe_model = fe_model
        self.caption_model_path = caption_model_path
        self.tokenizer_path = tokenizer_path
        self.max_length_path = max_length_path
        self.caption_model = None
        self.tokenizer = None
        self.max_length = None
        self.load_model()

    def load_model(self):
        # Load tokenizer and max length
        with open(self.tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        with open(self.max_length_path, 'r') as f:
            self.max_length = int(f.read())
        # Load captioning model
        self.caption_model = tf.keras.models.load_model(self.caption_model_path)

    def read_image(self, path, img_size=224):
        img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
        img = img_to_array(img)
        img = img / 255.0
        return img

    def extract_features(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        feature = self.fe_model.predict(img, verbose=0)
        return feature

    def idx_to_word(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_caption(self, image_path):
        feature = self.extract_features(image_path)
        in_text = "startseq"
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            y_pred = self.caption_model.predict([feature, sequence], verbose=0)
            y_pred = np.argmax(y_pred)
            word = self.idx_to_word(y_pred)
            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        return in_text

# Define the feature extraction model
fe = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, pooling='avg')

# Initialize the captioning model instance
caption_model_instance = ImageCaptioningModel(fe, 'caption_model_path', 'tokenizer.pkl', 'max_length.txt')

class ImageApi:
    class Upload(Resource):
        def post(self):
            try:
                data = request.get_json()
                image_data = data.get('image')

                conn = sqlite3.connect('sqlite.db')
                cursor = conn.cursor()

                cursor.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, image_data BLOB)''')
                cursor.execute("INSERT INTO images (image_data) VALUES (?)", (image_data,))
                conn.commit()
                conn.close()

                return jsonify({'message': 'Image uploaded successfully'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    class Predict(Resource):
        def post(self):
            try:
                data = request.get_json()
                url = data["image_data"]
                with urllib.request.urlopen(url) as response:
                    with open('temp.jpg', 'wb') as f:
                        f.write(response.read())

                image_path = 'temp.jpg'
                caption = caption_model_instance.predict_caption(image_path)

                return jsonify({'caption': caption})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

# Add resources to API
api.add_resource(ImageApi.Upload, '/upload')
api.add_resource(ImageApi.Predict, '/predict')
