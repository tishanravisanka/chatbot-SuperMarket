from flask import Flask, redirect, url_for, render_template,request
import nltk
from nltk.stem.lancaster import LancasterStemmer # stemming algorithm
import numpy
import tflearn
from tensorflow.python.framework import ops
import random
import json
import pickle

stemmer = LancasterStemmer()
app = Flask(__name__)

with  open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:

    words = [] # words in the pattern
    labels = [] # distinct tags

    # patterns and mapped tag
    docs_x = [] 
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] # stem nad remove ?
    words = sorted(list(set(words))) # remove duplicates
    labels = sorted(labels)


    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))] # make a empty list

    # go through each pattern and encode them
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc] # stem each word in the pattern

        # check in the main wordlist if it is exist in the docs_x map 0,1
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]

        output_row[labels.index(docs_y[x])] = 1 #mark tag type as 1 accoding to the soted tags

        training.append(bag)

        output.append(output_row)


    # convert to np array
    training = numpy.array(training) # map 0,1 according to the sorted words list for each pattern
    output = numpy.array(output)    # map 0,1 according to the sorted tag for each pattern

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)



ops.reset_default_graph()
# building the nural network
net = tflearn.input_data(shape=[None, len(training[0])])
print("\n\n\n\n")
# two hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    # a
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

itemList = []
reponse = []
req = []

def chat(usrMessage):
    # print("Start talking with the bot (type quit to stop)!")
    while True:
        # inp = input("You: ")
        inp = usrMessage
        if inp.lower() == "quit":
            break
        responses = []
        results = []
        results = model.predict([bag_of_words(inp, words)]) # give probability of each word
        results_index = numpy.argmax(results) # give the gratest value index in the list
        if(results[0][results_index]>0.7):
            # match the tag of the data
            tag = labels[results_index]
            # get all apropriate responces according to the tag
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            # select a random response
            if(RepresentsInt(random.choice(responses).split()[-1]) ):
                itemList.append(tag+" are in Shelf "+random.choice(responses).split()[-1])
            return (random.choice(responses))

        else:
            return ("I didn't get it. Please try a different question")

# home
@app.route("/")
def home():
    global reponse
    global req
    global itemList
    reponse = []
    req = []
    itemList = []
    return render_template("home.html",reponse=reponse,req=req)

# chat
@app.route('/result',methods = ['POST', 'GET'])
def result():
    form = request.form
    if request.method == 'POST':
        reponse.append(chat(request.form['msg']))
        req.append(request.form['msg'])

    return render_template("home.html",req=req,reponse =reponse)
    # return render_template("home.html",message = result)

# cart
@app.route('/cart',methods = ['POST', 'GET'])
def cart():
    return render_template("cart.html",list=itemList)


if __name__ =="__main__":
    app.run(debug=True)


chat("usrMessage")
print(itemList)