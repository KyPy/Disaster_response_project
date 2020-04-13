# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:18:39 2020

@author: kaisa
"""


from sqlalchemy import create_engine
import matplotlib.pyplot as plt

import os
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filename):
    
    # load data
    engine = create_engine('sqlite:///' + database_filename)
    
    
    df = pd.read_sql_table('f8_disater_response_data', engine)
    
    X = df['message']
    y = df.drop(columns=['id', 'message','original','genre'])
    
    return X, y

def tokenize(text):
    
    # remove urls
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





def display_results(y_test, y_pred):
    
    y_pred = pd.DataFrame(data=y_pred, columns=y_test.columns)
    
    # confusion matrix
    confusion_mat = np.zeros((2,2))
    
    for c in y_test.columns:
        confusion_mat += confusion_matrix(y_test[c], y_pred[c])
    
    confusion_mat = np.array(confusion_mat, dtype=np.float) / np.sum(confusion_mat)
    
    # accuracy
    accuracy = {c: (y_test[c]==y_pred[c]).mean() for c in y_test.columns}
    
    
    print("Confusion Matrix:\n", confusion_mat)
    print("Scores:")
    
    scores = {'f1': f1_score, 'precision': precision_score, 'recall': recall_score}
    for c in y_test.columns:
        t = '{:25}'.format(c+':') 
        for name, score in scores.items():
            try:
                t += '{}={:.2f}  '.format(name, score(y_test[c], y_pred[c])) #, zero_division=1
            except:
                t += 'error'
                pass
        print(t)
        
        
#    for c, v in accuracy.items():
#        print('    {}: {:.2f}'.format(c, v))

def check_recall(y_train, y_test, y_pred):
    
    recall = [recall_score(y_test[c], y_pred[c]) for c in y_test.columns]
    occurance = [(np.sum(y_test[c])+np.sum(y_train[c])) / (len(y_test)+len(y_train))  for c in y_test.columns]
    labels = y_test.columns.tolist()
    
    labels = [x for _,x in sorted(zip(recall,labels))]
    occurance = [x for _,x in sorted(zip(recall,occurance))]
    recall.sort()
    
    return recall, occurance, labels



X, y = load_data('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\Disaster_response_project\\data\\database.db')
X_train, X_test, y_train, y_test = train_test_split(X, y)

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

parameters = {'clf__estimator__n_estimators':[50, 100],
              'clf__estimator__criterion': ['gini','entropy']
             }
model = GridSearchCV(pipeline, parameters, n_jobs=-4)

# train classifier
model.fit(X_train, y_train)

# predict on test data
y_pred = model.predict(X_test)

# display results
display_results(y_test.reset_index(drop=True), y_pred)

y_pred = pd.DataFrame(data=y_pred, columns=y_test.columns)


recall, occurance, labels = check_recall(y_train, y_test.reset_index(drop=True), y_pred)


fig,ax = plt.subplots(nrows=2, sharex=True)
ax[0].bar(x=range(len(recall)), height=recall)
ax[1].bar(x=range(len(recall)), height=occurance)
ax[1].set_xticks(range(len(labels)))
ax[1].set_xticklabels(labels, rotation=90)
fig.tight_layout()







