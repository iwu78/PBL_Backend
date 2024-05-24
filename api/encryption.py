from model.encryption import hide_message, get_message
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db, project_path
import base64
import os
import base64
from PIL import Image
import numpy as np
import io


steg_bp = Blueprint("steg", __name__, url_prefix='/api/steg')
steg_api = Api(steg_bp)

class StegAPI(Resource):
    def post(self):
        body = request.get_json()
        imagedata = base64.b64decode(body['image'])
        image_file = io.BytesIO(imagedata)
        image = Image.open(image_file)
        unraveled = np.asarray(image)
        input_array = list(unraveled.ravel())
        width, height = image.size
        message = body['message']
        newImage = hide_message(message, input_array)  # Assuming hide_message returns a list/array of pixel data

        # Assuming newImage needs to be converted back to an image
        encoded_array = np.array(newImage, dtype=np.uint8).reshape(unraveled.shape)
        encoded_image = Image.fromarray(encoded_array)

        # Convert the PIL Image to bytes and then encode to base64 for JSON transmission
        img_byte_arr = io.BytesIO()
        encoded_image.save(img_byte_arr, format='PNG')  # You can change 'PNG' to another format if needed
        encoded_image_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        return jsonify({'image': encoded_image_data})
        


# Add resources outside the class definition
steg_api.add_resource(StegAPI, '/')
# images_api.add_resource(PostImagesAPI, '/upload')
