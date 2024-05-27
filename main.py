import threading

# import "packages" from flask
from flask import render_template,request  # import render_template from "public" flask libraries
from flask.cli import AppGroup


# import "packages" from "this" project
from __init__ import app, db, cors  # Definitions initialization


# setup APIs
from api.covid import covid_api # Blueprint import api definition
from api.joke import joke_api # Blueprint import api definition
from api.user import user_api # Blueprint import api definition
from api.player import player_api # teacher example API definition
from api.titanic import titanic_api # Group Machine Learning Project API definition
from api.collaborapost import post_api # MY PERSONAL API DEFINITION
from api.concussion import concussion_api # Group Machine Learning Project API definition
from api.songs import Song_api # Group Partner API definition
from api.model import model_api 
from api.image import images_bp
from api.encryption import steg_bp
# from api.stockMLapi import stock_api

# Teammate database migrations
from model.users import initUsers
from model.players import initPlayers
from model.titanicML import initTitanic
from model.concussion import initConcussion
from model.songs import initSongs
from model.images import initEasyImages

# setup App pages
from projects.projects import app_projects # Blueprint directory import projects definition


# Initialize the SQLAlchemy object to work with the Flask app instance
db.init_app(app)

# register URIs
app.register_blueprint(steg_bp)
app.register_blueprint(joke_api) # register teacher api routes
app.register_blueprint(covid_api) # register teacehr api routes
app.register_blueprint(user_api) # register teacher api routes
app.register_blueprint(player_api) # register teacher api routes
app.register_blueprint(titanic_api) # register teacher api routes
app.register_blueprint(app_projects) # register teacher app pages
app.register_blueprint(post_api) #registering my personal API
app.register_blueprint(concussion_api) #registering Group ML api
app.register_blueprint(Song_api)
app.register_blueprint(model_api)
app.register_blueprint(images_bp)
# app.register_blueprint(stock_api)


@app.errorhandler(404)  # catch for URL not found
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

@app.route('/')  # connects default URL to index() function
def index():
    return render_template("index.html")

@app.route('/table/')  # connects /stub/ URL to stub() function
def table():
    return render_template("table.html")

# Create an AppGroup for custom commands
custom_cli = AppGroup('custom', help='Custom commands')

# Define a command to generate data
@custom_cli.command('generate_data')
def generate_data():
    initUsers()
    initPlayers()
    initTitanic()
    initConcussion()
    initEasyImages()


@app.before_request
def before_request():
    initSongs()
    allowed_origin = request.headers.get('Origin')
    if allowed_origin in ['http://localhost:4100', 'http://127.0.0.1:4100', 'https://nighthawkcoders.github.io']:
        cors._origins = allowed_origin
# Register the custom command group with the Flask application
app.cli.add_command(custom_cli)
        
# this runs the application on the development server
if __name__ == "__main__":
    # change name for testing
    app.run(debug=True, host="0.0.0.0", port="8086")
