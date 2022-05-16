import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout

lemm = WordNetLemmatizer()
intsjs = json.loads(open('intents.json').read())
ignore_letters = ['?', '!', '.', ',']
words = []
diffcs = []
docs = []

for intent in intsjs['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        docs.append((word_list, intent['tag']))
        if intent['tag'] not in diffcs:
            diffcs.append(intent['tag'])

words = [lemm.lemmatize(word) for word in words if word not in ignore_letters]
diffcs = sorted(set(diffcs))
words = sorted(set(words))

pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(diffcs,open('diffcs.pkl', 'wb'))

teach = []
output = [0] * len(diffcs)

for doc in docs:
    bag = []
    pattern = doc[0]
    pattern = [lemm.lemmatize(word.lower()) for word in pattern]
    for word in words:
        bag.append(1) if word in pattern else bag.append(0)

    output_row = list(output)
    output_row[diffcs.index(doc[1])] = 1
    teach.append([bag, output_row])

random.shuffle(teach)
teach = np.array(teach)

teach1 = list(teach[:, 0])
teach2 = list(teach[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(teach1[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(teach2[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

modeling = model.fit(np.array(teach1), np.array(teach2), epochs=200, batch_size=5, verbose=1)
model.save('cbmodel.h5', modeling)
print("Done")