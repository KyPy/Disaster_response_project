# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:18:39 2020

@author: kaisa
"""


from sqlalchemy import create_engine

import matplotlib.pyplot as plt

import io
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

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Input, Bidirectional, GRU, Convolution1D, GlobalMaxPool1D
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras import layers, models, optimizers
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
    
    return ' '.join(clean_tokens)

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

#y = y[['hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'fire']]

X = [tokenize(text) for text in X]

X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#pipeline = Pipeline([
#    ('vect', CountVectorizer(tokenizer=tokenize)),
#    ('tfidf', TfidfTransformer())
#])
#
## train classifier
#res = pipeline.fit_transform(X_train, y_train)


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return data



# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\data\\wiki-news-300d-1M.vec')):
    if i!=0:
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

embeddings_index = load_vectors('D:\\Datensicherung\\Projekte\\Udacity_DataScience\\data\\wiki-news-300d-1M.vec')


# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(X_train)
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=50)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=50)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector




def create_rcnn(X):
    
    model = Sequential()
    
    

    # Add an Input Layer
    #input_layer = layers.Input((70, ))
    #model.add(Input((X.shape[1], )))

    # Add the word embedding Layer
    #embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    #embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
    model.add(Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.3))
    
    # Add the recurrent layer
    #rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)
    model.add(Bidirectional(GRU(50, return_sequences=True)))
    
    # Add the convolutional Layer
    #conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)
    model.add(Convolution1D(100, 3, activation="relu"))

    # Add the pooling Layer
    #pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
    model.add(GlobalMaxPool1D())

    # Add the output Layers
    #output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    #output_layer1 = layers.Dropout(0.25)(output_layer1)
    #output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(36, activation="sigmoid")) #36
    

    # Compile the model
    #model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

epochs = 5
batch_size = 64

model = create_rcnn(train_seq_x)
history = model.fit(train_seq_x, y_train, epochs=epochs, batch_size=batch_size, validation_data=(valid_seq_x, y_test))



plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(valid_seq_x)
y_pred = np.round(y_pred)
#y_pred[y_pred>=0.3] = 1
#y_pred[y_pred<0.3] = 0

y_pred = pd.DataFrame(data=y_pred, columns=y_test.columns)
y_pred.reset_index(inplace=True, drop=True)

display_results(y_test.reset_index(drop=True), y_pred)


recall, occurance, labels = check_recall(y_train, y_test.reset_index(drop=True), y_pred)


fig,ax = plt.subplots(nrows=2, sharex=True)
ax[0].bar(x=range(len(recall)), height=recall)
ax[1].bar(x=range(len(recall)), height=occurance)
ax[1].set_xticks(range(len(labels)))
ax[1].set_xticklabels(labels, rotation=90)
fig.tight_layout()