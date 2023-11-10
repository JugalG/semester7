import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json

with open("intents.json") as file:
    data = json.load(file)

stemmer = LancasterStemmer()

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize and stem the words in the pattern
        wrds = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(pattern)]
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    # Store unique intent tags as labels
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stem and sort the words
words = sorted(list(set([stemmer.stem(w) for w in words if w != "?"])))
labels = sorted(labels)

# Create training and output data
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = [1 if w in doc else 0 for w in words]

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Reset the TensorFlow graph
tf.reset_default_graph()

# Define the neural network model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Create and train the model
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

# Save the trained model
model.save("model.tflearn")





def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        confidence = results[0][results_index]

        if confidence > 0.7:
            tag = labels[results_index]
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I didn't understand that. Please try again.")
chat()
