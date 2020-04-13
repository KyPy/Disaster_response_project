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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))


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



X, y = load_data('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\Disaster_response_project\\data\\database.db')
#X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#pipeline = Pipeline([
#    ('vect', CountVectorizer(tokenizer=tokenize)),
#    ('tfidf', TfidfTransformer())
#])
#
## train classifier
#res = pipeline.fit_transform(X_train, y_train)


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)



model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(36, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(X_test)

y_pred = pd.DataFrame(data=np.round(y_pred), columns=Y_test.columns)
y_pred.reset_index(inplace=True, drop=True)

display_results(Y_test.reset_index(drop=True), y_pred)






