# ! pip install stemmer
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
with open('./data/arabic.json') as json_data:
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


# create training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# reset underlying graph data

# ! pip install tflearn 


import tflearn
# tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=500, batch_size=10, show_metric=True)
model.save('model.tflearn')

# essay greading -- text  NLP 

ERROR_THRESHOLD = 0.25

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag %s" % w)
    return np.array(bag)
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first results
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return print(random.choice(i['responses']))

            results.pop(0)
def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first results
                if i['tag'] == results[0][0]:
                  if 'context_set' in i :
                    if show_details:
                      print('context:',i['context_set'])
                      context[userID]=i['context_set']
                  if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter']== context[userID]):
                    if show_details :
                      print('tag:',i['tag'])
                    # a random response from the intent
                    return (random.choice(i['responses']))

            results.pop(0)

print("Classera ChatBot:My name is Shahed, What do you want to chat about? Please fill the input that you search about ")
print("If you want to exit any time just say bye")
flag=True
while flag:
  ask=input('User : ')
  print('Classera Bot: ' , response(ask))
  if ask == "شكرا"   or ask == "باي":
    flag=False