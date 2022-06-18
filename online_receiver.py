import argparse
import csv
import socket
import time
import json
import numpy
import threading


from keras.layers import Activation, Dense
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from datetime import timezone
import datetime

import matplotlib.pyplot as plt


# Command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--port', dest='serverPort', type=int, default=9000, help='port of the server that is doing online training')

args = parser.parse_args()

inputDataQueue = []
outputDataQueue = []
testDataQueue = []

predictions = []
real = []

xResults = []
yResults = []

nnVersion = 0

optimizer = Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

numberVariables = 55
numberOutputs = 3

L1Model = Sequential()

L1Model.add(Dense(10, input_shape=(numberVariables,) ,activation='tanh'))
L1Model.add(Dense(5, activation='tanh'))
L1Model.add(Dense(2, activation='softmax'))

L1Model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy', 'categorical_crossentropy'])


L2Model = Sequential()

L2Model.add(Dense(10, input_shape=(numberVariables,) ,activation='relu'))
L2Model.add(Dense(5, activation='relu'))
L2Model.add(Dense(numberOutputs, activation='softmax'))

L2Model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy', 'categorical_crossentropy'], run_eagerly=True)

def DataReceiver():
    global inputDataQueue
    global outputDataQueue
    global testDataQueue

    global real

    connection, address = trainingSocket.accept()

    while True:
        message = json.loads(connection.recv(1024).decode())

        inputDataQueue.append(message)
        testDataQueue.append(message)

        message = json.loads(connection.recv(1024).decode())

        outputDataQueue.append(message)
        real.append(message)

def NetworkTrainer():
    global inputDataQueue
    global outputDataQueue
    global L1Model
    global L2Model

    global nnVersion

    while True:
        with open('results_online_training.txt', 'a') as outFile:
            if len(inputDataQueue) > 0 and len(outputDataQueue) > 0:
                print(len(inputDataQueue), len(outputDataQueue))

                sequenceId = outputDataQueue[0][0]
                timestamp = outputDataQueue[0][1]
                x = numpy.asarray(inputDataQueue[0][2]).astype(float).reshape((1, numberVariables))
                y = numpy.asarray(outputDataQueue[0][2]).astype(float)

                #print(x,y)

                del inputDataQueue[0]
                del outputDataQueue[0]

                L1Y = None
                L2Y = None
                if numpy.argmax(y) > 0:
                    L1Y = numpy.asarray(to_categorical(1, 2)).reshape((1, 2))
                    L2Y = numpy.asarray(to_categorical(1, numberOutputs)).reshape((1, numberOutputs))

                    L1Model.fit(x=x, y=L1Y, batch_size=1, epochs=1)
                    L2Model.fit(x=x, y=L2Y, batch_size=1, epochs=1)
                else:
                    L1Y = numpy.asarray(to_categorical(0, 2)).reshape((1, 2))
                    L1Model.fit(x=x, y=L1Y, batch_size=1, epochs=1)

                currentTimestamp = datetime.datetime.now(timezone.utc).replace(tzinfo=timezone.utc).timestamp()
                outFile.write(str(sequenceId)+","+str(currentTimestamp-timestamp)+"\n")

                #print(len(inputDataQueue), len(outputDataQueue))

                nnVersion += 1

trainingSocket = socket.socket()
trainingSocket.bind(('', args.serverPort))
trainingSocket.listen(5)

errorCount = 0
count = 0
while True:
    receiverThread = threading.Thread(target=DataReceiver, args=())
    receiverThread.start()

    trainingThread = threading.Thread(target=NetworkTrainer, args=())
    trainingThread.start()

    while True:
        with open('results_online_test.txt', 'a') as outFile:
            if len(testDataQueue) > 0:
                timestamp = testDataQueue[0][1]
                x = numpy.asarray(testDataQueue[0][2]).astype(float).reshape((1, numberVariables))
                del testDataQueue[0]

                auxiliaryL1Model = keras.models.clone_model(L1Model)
                auxiliaryL1Model.build((None, numberVariables))
                auxiliaryL1Model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy', 'categorical_crossentropy'])
                auxiliaryL1Model.set_weights(L1Model.get_weights())

                auxiliaryL2Model = keras.models.clone_model(L2Model)
                auxiliaryL2Model.build((None, numberVariables))
                auxiliaryL2Model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy', 'categorical_crossentropy'], run_eagerly=True)
                auxiliaryL2Model.set_weights(L2Model.get_weights())

                prediction = auxiliaryL1Model.predict(x)
                if numpy.argmax(prediction[0]) > 0:
                    prediction = auxiliaryL2Model.predict(x)

                currentTimestamp = datetime.datetime.now(timezone.utc).replace(tzinfo=timezone.utc).timestamp()
                predictions.append([currentTimestamp-timestamp, prediction[0]])

                #print('Version', nnVersion)

                while len(predictions) > 0 and len(real) > 0:
                    timestampDifference = predictions[0][0]
                    predicted = numpy.argmax(predictions[0][1])

                    sequenceId = real[0][0]
                    actual = numpy.argmax(numpy.asarray(real[0][2]).astype(float))

                    del predictions[0]
                    del real[0]

                    if predicted != actual:
                        errorCount += 1

                    outFile.write(str(sequenceId)+","+str(timestampDifference)+","+str(errorCount)+"\n")

                    xResults.append(count)
                    yResults.append(errorCount)
                    count += 1

                    #plt.plot(xResults, yResults)
                    #plt.draw()
                    #plt.pause(0.001)
