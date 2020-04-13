import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


app = Flask(__name__)

def tokenize(text):
    """
    Description: This function cleans and tokenizes text messages.
                 Cleaning steps:
                     - replace url by placeholder
                     - Normalize text to lower case
                     - Tokenize text into words
                     - Remove stop words
                     - Lemmatize words

    Arguments:
        text:  string

    Returns:
        list of clean tokens
    """
    
    # remove urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('f8_disater_response_data', engine)
labels = df.drop(columns=['id', 'message','original','genre']).columns.tolist()
df_ana = pd.read_csv('../models/model_analysis.csv')

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    classes_count = df.drop(columns=['id', 'message','original','genre']).sum()
    
    words = [re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split() for text in df['message']]
    words = [item for sublist in words for item in sublist]
    word_counts = pd.Series(words).value_counts()
    
    common_words = []
    word_count_values = []
    i=0
    while len(common_words)<10:
        if word_counts.index[i] not in stopwords.words("english"):
            common_words.append(word_counts.index[i])
            word_count_values.append(word_counts.iloc[i])
        i+=1
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=list(classes_count.index),
                    y=list(classes_count/len(df)*100)
                )
            ],

            'layout': {
                'title': 'Occurance of classes in training set',
                'yaxis': {
                    'title': "Percentage[%]"
                },
                'xaxis': {
                    'title': "Class"
                },
                'margin': {'b': 160}
            }
        },
        {
            'data': [
                Bar(
                    x=common_words,
                    y=word_count_values
                )
            ],

            'layout': {
                'title': 'Most common words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "word"
                }
            }
        }
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    for i in range(len(classification_labels)):
        print(classification_labels[i], labels[i])
    classification_results = dict(zip(labels, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()