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
from api.player import player_api
from api.titanic import titanic_api
<<<<<<< HEAD
from api.collaborapost import post_api
from api.stockMLapi import stock_api
=======
from api.concussion import concussion_api
from api.songs import Song_api
>>>>>>> fd98e8d92bd76846ac16bfc5f6e635a330ff9c0d
# database migrations
from model.users import initUsers
from model.players import initPlayers
from model.titanicML import initTitanic
from model.concussion import initConcussion
from model.songs import initSongs

# setup App pages
from projects.projects import app_projects # Blueprint directory import projects definition


# Initialize the SQLAlchemy object to work with the Flask app instance
db.init_app(app)

# register URIs
app.register_blueprint(joke_api) # register api routes
app.register_blueprint(covid_api) # register api routes
app.register_blueprint(user_api) # register api routes
app.register_blueprint(player_api)
app.register_blueprint(titanic_api) # register api routes
app.register_blueprint(app_projects) # register app pages
<<<<<<< HEAD
app.register_blueprint(post_api)
app.register_blueprint(stock_api)
=======
app.register_blueprint(concussion_api)
app.register_blueprint(Song_api)

>>>>>>> fd98e8d92bd76846ac16bfc5f6e635a330ff9c0d

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

@app.before_request
def before_request():
    initSongs()

# Register the custom command group with the Flask application
app.cli.add_command(custom_cli)
        
# this runs the application on the development server
if __name__ == "__main__":
    # change name for testing
    app.run(debug=True, host="0.0.0.0", port="8086")
