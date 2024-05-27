from model.encryption import hide_message, get_message, imageToBase64, base64toImage
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db, project_path
import base64
import os
import base64
from PIL import Image
import numpy as np
import io


steg_bp = Blueprint("steg_api", __name__, url_prefix='/api/steg')
steg_api = Api(steg_bp)

class StegAPI(Resource):
    def post(self):
        body = request.get_json()
        image = base64toImage(body['image'])
        unraveled = np.asarray(image)
        input_array = list(unraveled.ravel())
        message = body['message']
        newImage, bit_count, values_modified = hide_message(message, input_array)

        # Assuming newImage needs to be converted back to an image
        encoded_array = np.array(newImage, dtype=np.uint8).reshape(unraveled.shape)
        encoded_image = Image.fromarray(encoded_array)

        return jsonify({'image': imageToBase64(encoded_image), 'bits_used': bit_count, 'values_modified': values_modified})
    
    def put(self):
        body = request.get_json()
        image = base64toImage(body['image'])
        unraveled = np.asarray(image)
        input_array = list(unraveled.ravel())
        message = get_message(input_array)
        if message:
            return jsonify({'message': message})
        else:
            return {'message': 'No message found in image!'}, 210
        


# Add resources outside the class definition
steg_api.add_resource(StegAPI, '/')
# images_api.add_resource(PostImagesAPI, '/upload')
