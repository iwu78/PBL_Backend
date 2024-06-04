- Being a backend, this repository is dedicated to developing APIs and necessary models to host data collection, a key example of this is databases. We have connected the Frontend and Backend via the concept of fetching the backend. We fetch links for specific APIs based on their registered blueprint name and URL prefix.

### Steganography
Steganography is a technique for hiding secret messages within other non-secret data. This project explores the art of steganography, allowing users to encode and decode messages within images, text, or audio files for covert communication. With advanced encryption techniques and secure data handling, Steganography provides a secure and reliable platform for hidden message exchange.
### Collabora
Collabora was ideated to address certain struggles that we and fellow students have experienced in their academic career. With several tests on a timely basis, there are often questions that we forget to ask our teachers in school and are frantically trying to figure out the solution the night before. Collabora acts as a mediator between teachers or experts of the topic in which the user can post a question publiclyin which teachers or experts can also see and answer those questions, other students also have the ability to answer the question which promotes peer to peer communication which is essential.
### Stock ML
Using foundational Machine Learning concepts, given the SP500 datasets I have created an ML model that once trained can be used to predict the viability of an investment and the probability that your money will increase in terms of the Stock. Created a UI which asks users for the opening price, closing price, highest price, high price, low price to effectively predict the viability of the users investment.



### Spotify 
The Spotify Database is a comprehensive collection of music metadata, including artists, albums, tracks, and genres. Users can explore and discover their favorite music and artists with ease using powerful search and filtering functionalities. With access to a vast library of music information, the Spotify Database enriches the music discovery experience for users.

Key Features: play the music from the backend.

Users select a song which is then fetched from the backend and played through the computers speakers.

Interface is designed to be verys similar to the real spotify.

### Concussion Recovery Prediction Project
This project aims to provide personalized recovery recommendations for individuals experiencing symptoms following a concussion. The core functionality involves a web-based form where users can input the severity of their symptoms. The backend processes this data to generate a total score and specific recommendations based on the symptom severity.

Key Features:
Symptom Input Form:

Users select the severity of various symptoms (e.g., Headache, Neck Pain, Nausea) from a predefined list.
The form is dynamically generated and organized into three columns for better user experience.
Symptom Scoring:

Each symptom has an associated weight, and the severity is mapped to a numerical value.
The total score is calculated by multiplying the symptom weight by the severity value.
The total score and individual symptom scores are used to determine the severity of the condition.
Recommendations Generation:

Based on the scores, the system generates specific recommendations to aid in recovery.
Recommendations are tailored to the severity of each symptom and include general advice as well as medical guidance for more severe cases.
The recommendations are displayed below the form upon submission.
User Interface:

The interface is designed to be user-friendly and visually appealing, with a clean layout and responsive design.
Background and styling elements provide a calming and professional appearance, enhancing the overall user experience.
This project combines frontend user interaction with backend data processing to deliver a helpful tool for individuals recovering from a concussion. The system leverages medical guidelines and best practices to offer actionable advice, supporting users in their recovery journey.

# Flask Portfolio Starter

Use this project to create a Flask Servr.

Runtime link: <https://flask.nighthawkcodingsociety.com/>
GitHub link: https://github.com/nighthawkcoders/flask_portfolio

## Conventional way to get started

> Quick steps that can be used with MacOS, WSL Ubuntu, or Ubuntu; this uses Python 3.9 or later as a prerequisite.

- Open a Terminal, clone project and cd to project area

```bash
mkdir ~/vscode; cd ~/vscode

git clone https://github.com/nighthawkcoders/flask_portfolio.git

cd flask_portfolio
```

- Install python dependencies for Flask, etc.

```bash
pip install -r requirements.txt
```

- Run from Terminal without VSCode

  - Setup database and init data
  
  ```bash
    ./migrate.sh
    ```

  - Run python server from command line without VSCode

    ```bash
    python main.py
    ```

### Open project in VSCode

- Prepare VSCode and run

  - From Terminal run VSCode

    ```bash
    code .
    ```

  - Open Setting: Ctl-Shift P or Cmd-Shift
    - Search Python: Select Interpreter
    - Match interpreter to `which python` from terminal

  - Select main.py and Play button
  - Try Play button and try to Debug

## Idea

> The purpose of project is to serve APIs.  It is the backend piece of a Full-Stack project.  Review `api` folder in project for endpoints.

### Visual thoughts

> The Starter code should be fun and practical.

- Organize with Bootstrap menu
- Add some color and fun through VANTA Visuals (birds, halo, solar, net)
- Show some practical and fun links (hrefs) like Twitter, Git, Youtube
- Build a Sample Page (Table)
- Show project specific links (hrefs) per page

### Files and Directories in this Project

These are some of the key files and directories in this project

README.md: This file contains instructions for setting up the necessary tools and cloning the project. A README file is a standard component of all properly set up GitHub projects.

requirements.txt: This file lists the dependencies required to turn this Python project into a Flask/Python project. It may also include other backend dependencies, such as dependencies for working with a database.

main.py: This Python source file is used to run the project. Running this file starts a Flask web server locally on localhost. During development, this is the file you use to run, test, and debug the project.

Dockerfile and docker-compose.yml: These files are used to run and test the project in a Docker container. They allow you to simulate the project’s deployment on a server, such as an AWS EC2 instance. Running these files helps ensure that your tools and dependencies work correctly on different machines.

instances: This directory is the standard location for storing data files that you want to remain on the server. For example, SQLite database files can be stored in this directory. Files stored in this location will persist after web application restart, everyting outside of instances will be recreated at restart.

static: This directory is the standard location for files that you want to be cached by the web server. It is typically used for image files (JPEG, PNG, etc.) or JavaScript files that remain constant during the execution of the web server.

api: This directory contains code that receives and responds to requests from external servers. It serves as the interface between the external world and the logic and code in the rest of the project.

model: This directory contains files that implement the backend functionality for many of the files in the api directory. For example, there may be files in the model directory that directly interact with the database.

templates: This directory contains files and subdirectories used to support the home and error pages of the website.

.gitignore: This file specifies elements to be excluded from version control. Files are excluded when they are derived and not considered part of the project’s original source. In the VSCode Explorer, you may notice some files appearing dimmed, indicating that they are intentionally excluded from version control based on the rules defined in .gitignore.

### Implementation Summary

#### July 2023

> Updates for 2023 to 2024 school year.

- Update README with File Descriptions (anatomy)
- Add JWT and add security features to data
- Add migrate.sh to support sqlite schema and data upgrade

#### January 2023

> This project focuses on being a Python backend server.  Intentions are to only have simple UIs an perhaps some Administrative UIs.

#### September 2021

> Basic UI elements were implemented showing server side Flask with Jinja 2 capabilities.

- Project entry point is main.py, this enables Flask Web App and provides capability to renders templates (HTML files)
- The main.py is the  Web Server Gateway Interface, essentially it contains a HTTP route and HTML file relationship.  The Python code constructs WSGI relationships for index, kangaroos, walruses, and hawkers.
- The project structure contains many directories and files.  The template directory (containing html files) and static directory (containing js files) are common standards for HTML coding.  Static files can be pictures and videos, in this project they are mostly javascript backgrounds.
- WSGI templates: index.html, kangaroos.html, ... are aligned with routes in main.py.
- Other templates support WSGI templates.  The base.html template contains common Head, Style, Body, Script definitions.  WSGI templates often "include" or "extend" these templates.  This is a way to reuse code.
- The VANTA javascript statics (backgrounds) are shown and defaulted in base.html (birds), but are block replaced as needed in other templates (solar, net, ...)
- The Bootstrap Navbar code is in navbar.html. The base.html code includes navbar.html.  The WSGI html files extend base.html files.  This is a process of management and correlation to optimize code management.  For instance, if the menu changes discovery of navbar.html is easy, one change reflects on all WSGI html files.
- Jinja2 variables usage is to isolate data and allow redefinitions of attributes in templates.  Observe "{% set variable = %}" syntax for definition and "{{ variable }}" for reference.
- The base.html uses combination of Bootstrap grid styling and custom CSS styling.  Grid styling in observe with the "<Col-3>" markers.  A Bootstrap Grid has a width of 12, thus four "Col-3" markers could fit on a Grid row.
- A key purpose of this project is to embed links to other content.  The "href=" definition embeds hyperlinks into the rendered HTML.  The base.html file shows usage of "href={{github}}", the "{{github}}" is a Jinja2 variable.  Jinja2 variables are pre-processed by Python, a variable swap with value, before being sent to the browser.
