import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
import pandas as pd


model_api = Blueprint('model_api', __name__,
                   url_prefix='/api/titanic')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(model_api)
class TitanicAPI:        
    class _Titanic(Resource):
        def post(self):
            ## get body and data from frontend
            body = request.get_json()
            name = body.get('name')
            pclass = body.get('pclass')
            sex = body.get('sex')
            age = body.get('age')
            fmem = body.get('fmem')
            fare = body.get('fare')
            embark = body.get('embark')
            ## create new dataframe object for model to use
            passenger = pd.DataFrame({
                'name': [name],
                'pclass': [pclass],
                'sex': [sex],
                'age': [age],
                'sibsp': [fmem], 
                'parch': [fmem], 
                'fare': [fare], 
                'embarked': [embark], 
                ## alone can be determined from number of family members and siblings
                'alone': [True if fmem == 0 else False]
            })
            ## run titanic
            initTitanic()
            ## predict survival
            return predictSurvival(passenger)
    api.add_resource(_Titanic, '/')
    