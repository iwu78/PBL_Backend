from model.images import Images
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db, project_path
import base64
import os

images_bp = Blueprint("images", __name__, url_prefix='/api/images')
images_api = Api(images_bp)


class ImagesAPI(Resource):
    def get(self):
        images = Images.query.all()  # Assuming this gets all image records
        json_data_list = []

        for image in images:
            # Check if imageData is not None
            if image.imageData:
                # Convert the binary data to a base64 encoded string
                encoded_string = base64.b64encode(image.imageData).decode('utf-8')
                json_data_list.append({"imagePath": image.imagePath, "imageData": encoded_string})
            else:
                # Handle the case where imageData is None
                json_data_list.append({"error": f"Image data not found for image id {image.id}"})

        return jsonify(json_data_list)

class PostImagesAPI(Resource):
    def post(self):
        json_data = request.get_json()
        if "base64_string" in json_data and "name" in json_data:
            base64_string = json_data["base64_string"]
            name = json_data["name"]
            image_data = base64.b64decode(base64_string)
            # Save the image to the database
            image = Images(imagePath=os.path.join('images', f"{name}.jpg"), imageData=image_data)
            db.session.add(image)
            db.session.commit()
            return jsonify({"message": "Image saved successfully"})
        else:
            return jsonify({"error": "Invalid request"})

# Add resources outside the class definition
images_api.add_resource(ImagesAPI, '/')
images_api.add_resource(PostImagesAPI, '/upload')
            

        

            
    
    