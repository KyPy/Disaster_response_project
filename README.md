# Disaster Response Pipeline Project
Analysis of Covid-19 datasets

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Web Application](#webapp)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Assuming the base installation of Ananconda, two additional packages are necessary: sqlalchemy, pandas, scikit-learn, nltk, flask
The code should run with no issues using Python versions 3.*.

If the deep learning models should be applied, the following additional package is necessary: keras

## Project Motivation<a name="motivation"></a>

The goal is to train a model, to classify messages for relevance following a disaster. The message will be categorized into 36 categories, e.g. for relevance and what kind of help is needed.
The model is implemented in a web app, which can be used to classify new text messages.


## File Descriptions <a name="files"></a>

The files are split into three folders

    - data
        - disaster_categories.csv: data with all the categories with (0 for yes, 1 for no)
        - disaster_messages.csv: data with the message texts
        - process_data.py: script to read, clean, and save data into a database
    - models
        - train_classifier.py: machine learning pipeline scripts to train and export a classifier
		- test_functions.py: script to develop the main model (RandomForestClassifier)
		- test_functions_lstm.py: script to develop the main model (LSTM)
		- test_functions_rcnn.py: script to develop the main model (Recurrent Convolutional Neural Network (RCNN))
    - app
        - run.py: Flask file to run the web application
        - folder templates contains html files for the web application
	- img: Images of web application and training results


## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@kai.sandmann/analysing-influences-on-covid-19-spread-in-major-countries-f74cd7a0f309).

## Web Application<a name="webapp"></a>

The app, stored in folder "app" can be installed and launched via flask.

On the start page of the app, a short overview of the training dataset is given.
Also a textbox for entering a new text message is displayed.

**_Screenshot 1_**
![results](img/WebApp1.png)

After classifying a text message the result for all classes is displayed.

**_Screenshot 2_**
![results](img/WebApp2.png)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing messages dataset to train my model