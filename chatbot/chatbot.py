import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemm = WordNetLemmatizer()
intsjs = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
diffcs = pickle.load(open('diffcs.pkl', 'rb'))
mod = load_model('cbmodel.h5')

def sentenceclean(sentence):
    sentences = nltk.word_tokenize(sentence)
    sentences = [lemm.lemmatize(word) for word in sentences]
    return sentences

def worded(sentence):
    sentences = sentenceclean(sentence)
    bag = [0] * len(words)
    for w in sentences:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def prediction(sentence):
    bow = worded(sentence)
    res = mod.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': diffcs[r[0]], 'probability': str(r[1])})
    return return_list

def respond(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot is running")
while True:
    message = input("")
    ints = prediction(message)
    res = respond(ints, intsjs)
    print(res)