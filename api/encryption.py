from model.encryption import hide_message, get_message, imageToBase64, base64toImage
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db, project_path
import os
import base64
from PIL import Image
import numpy as np
import io


steg_bp = Blueprint("steg_api", __name__, url_prefix='/api/steg')
steg_api = Api(steg_bp)

class StegAPI(Resource):
    def post(self):
        body = request.get_json() # Get the JSON data from the request body
        image = base64toImage(body['image']) # Convert the base64-encoded image to PIL image

        unraveled = np.asarray(image)  # Convert the image to a NumPy array
        input_array = list(unraveled.ravel()) # Convert the image into a list pixel values
        message = body['message'] # Get the message to hide from the body
        newImage, bit_count, values_modified = hide_message(message, input_array) # Hide the message in the pixel values using the hide_message function

        encoded_array = np.array(newImage, dtype=np.uint8).reshape(unraveled.shape) # Reshape the modified pixel values back into the original image shape
        encoded_image = Image.fromarray(encoded_array) # Convert the NumPy array back to a PIL Image


        return jsonify({'image': imageToBase64(encoded_image), 'bits_used': bit_count, 'values_modified': values_modified}) # return new base64 encoded image
    
    def put(self):
        body = request.get_json() # Get the JSON data from the request body
        image = base64toImage(body['image']) # convert base64 encoded image to PIL image
        unraveled = np.asarray(image) # convert image to numpy array
        input_array = list(unraveled.ravel()) # Convert the image into a list pixel values
        message = get_message(input_array)# Extract  hidden message from pixel values using the get_message function

        if message: 
            return jsonify({'message': message})# if message found return it to user
        else:
            return {'message': 'No message found in image!'}, 210 # if no message found
        


# Add resources outside the class definition
steg_api.add_resource(StegAPI, '/')
# images_api.add_resource(PostImagesAPI, '/upload')
