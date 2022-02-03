! pip install stemmer
import numpy as np
import nltk
nltk.download('punkt') # use punkt tokenizer
nltk.download('wordnet') # use the wordNet dictionary
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.stem.lancaster import LancasterStemmer
import string
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import random
import tensorflow as tf
stemmer = LancasterStemmer()
from nltk import word_tokenize  # Arabic root
from nltk.stem.isri import ISRIStemmer # Arabic root


import json
with open('/data/arabic.json') as json_data:
  intents=json.load(json_data)
import re
st = ISRIStemmer()
words =[]
classes=[]
documents =[]
ignore_words=['?']
wordsfilter=[]
# loop in each sentence in our intents patterns
for intent in intents['intents']:
  for pattern in intent['patterns']:
    # tokenize each word in the sentence
    w=nltk.word_tokenize(pattern)
    # add to doc in our corpus
    words.extend(w)
    documents.append((w,intent['tag']))
    # add to our classes list
    root = [print(st.stem(line)) for line in nltk.word_tokenize(pattern) ]
    # root = [wordsfilter.append(st.stem(a)) for a in nltk.word_tokenize(pattern) ]
    
    if intent['tag'] not in classes:
      classes.append(intent['tag'])
# print(wordsfilter)

# stem and lower the word and remove duplicates
words=[stemmer.stem(w.lower()) for w in words if w not in ignore_words]

# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# remove duplicates
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words",words )
