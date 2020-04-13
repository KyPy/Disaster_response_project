import sys
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report
import pickle





def load_data(database_filepath):
    """
    Description: This function connects to the database containing disaster response messages.
                 It splits the data into input and output data for a classification task.
                 It's xpected, that the database is in sqlite format and the tablename is f8_disater_response_data.

    Arguments:
        database_filepath:  string Filepath to the database file

    Returns:
        input pandas DataFrame
        output pandas DataFrame
        list of columns in output DataFrame
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table('f8_disater_response_data', engine)
    
    X = df['message']
    y = df.drop(columns=['id', 'message','original','genre'])
    
    return X, y, y.columns.tolist()


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


def build_model():
    """ Builds a NLP pipeline """
    
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    parameters = {'clf__estimator__n_estimators':[50, 100]
                 }
    model = GridSearchCV(pipeline, parameters) #, n_jobs=4
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description: This function prints a classification report

    Arguments:
        model:  fitted sklearn model
        X_test: pandas DataFrame with input of test data
        Y_test: pandas DataFrame with output of test data
        category_names: list of columns in output DataFrame
    """
    
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.values, np.array([x for x in Y_pred]), 
        target_names=category_names))
    


def save_model(model, model_filepath):
    """ stores the model as a pickle file """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        

if __name__ == '__main__':
    main()