import argparse
import csv
import socket
import time
import json
import numpy
import threading


from keras.layers import Activation, Input, Dense
from keras.models import Sequential, Model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from datetime import timezone
import datetime

import matplotlib.pyplot as plt


# Command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--port', dest='serverPort', type=int, default=9000, help='port of the server that is doing online training')
parser.add_argument('--memory-size', dest='memorySize', type=int, default=20, help='number of samples to keep for training')
parser.add_argument('--batch-size', dest='batchSize', type=int, default=5, help='set number of samples by batch')
parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='set number of epochs of training')

args = parser.parse_args()

lock = threading.Lock()

dataQueue = {}
testDataQueue = {}
predictions = {}

xResults = []
yResults = []

nnVersion = 0

optimizer = Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

numberVariables = 94
numberOutputs = 5

inputLayer = Input(shape=(numberVariables,))
h2Layer = Dense(3, activation='tanh')(inputLayer)
outputLayer = Dense(numberVariables, activation='tanh')(h2Layer)
L1Model = Model(inputs=inputLayer, outputs=outputLayer)

L1Model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

LsModel = Sequential()

LsModel.add(Dense(10, input_shape=(1,) ,activation='tanh'))
LsModel.add(Dense(5, activation='tanh'))
LsModel.add(Dense(2, activation='softmax'))

LsModel.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy', 'categorical_crossentropy'])

L2Model = Sequential()

L2Model.add(Dense(10, input_shape=(numberVariables,) ,activation='relu'))
L2Model.add(Dense(5, activation='relu'))
L2Model.add(Dense(numberOutputs, activation='softmax'))

L2Model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy', 'categorical_crossentropy'], run_eagerly=True)

def DataReceiver():
    global lock

    global dataQueue
    global testDataQueue
    global predictions

    connection, address = trainingSocket.accept()

    while True:
        recivedMessage = connection.recv(10240).decode()

        for message in recivedMessage.split('EOF')[:-1]: # Discard last message because it will be an empty string after the last EOF
            #print(message)
            fields = json.loads(message)

            sequenceId = fields[0]
            timestamp = fields[1]
            data = fields[2]
            dataType = fields[3]

            lock.acquire()

            if not sequenceId in dataQueue:
                dataQueue[sequenceId] = {}

            dataQueue[sequenceId][dataType] = { 'timestamp': timestamp, 'data': data }

            if dataType == 'in':
                testDataQueue[sequenceId] = { 'timestamp': timestamp, 'data': data }
            else:
                if not sequenceId in predictions:
                    predictions[sequenceId] = {}

                predictions[sequenceId]['real'] = { 'timestamp': timestamp, 'output': data }

            if len(dataQueue) > args.memorySize:
                oldestSequenceId = min(dataQueue.keys()) # Get oldest sequence ID (min dictionary key)
                del dataQueue[oldestSequenceId]

            lock.release()

def NetworkTrainer():
    global lock

    global dataQueue
    global L1Model
    global LsModel
    global L2Model

    global nnVersion

    while True:
        L1X = []
        LsX = []
        LsY = []
        L2X = []
        L2Y = []
        lock.acquire()

        for sequenceId in dataQueue:
            if 'in' in dataQueue[sequenceId] and 'out' in dataQueue[sequenceId]: # if already received expected output
                if numpy.argmax(dataQueue[sequenceId]['out']['data']) > 0:
                    L1X.append(dataQueue[sequenceId]['in']['data'])
                    LsY.append(to_categorical(1, 2))
                    L2X.append(dataQueue[sequenceId]['in']['data'])
                    L2Y.append(dataQueue[sequenceId]['out']['data'])
                else: # is a "non-activity" data, don't need to go through second level
                    L1X.append(dataQueue[sequenceId]['in']['data'])
                    LsY.append(to_categorical(0, 2))

        lock.release()

        if len(L1X) > 0:
            x = numpy.asarray(L1X).astype(float)

            L1Model.fit(x=x, y=x, batch_size=args.batchSize, epochs=args.epochs, verbose=0)

            L1Predictions = L1Model.predict(x, batch_size=1)
            for i in range(len(L1Predictions)):
                prediction = L1Predictions[i]
                real = x[i]

                mse = 0
                for j in range(len(prediction)):
                    mse += numpy.square(prediction[j]-real[j])

                mse /= len(prediction)
                LsX.append(mse)

            x = numpy.asarray(LsX).astype(float)
            y = numpy.asarray(LsY).astype(float)

            LsModel.fit(x=x, y=y, batch_size=args.batchSize, epochs=args.epochs, verbose=0)

        if len(L2X) > 0:
            x = numpy.asarray(L2X).astype(float)
            y = numpy.asarray(L2Y).astype(float)

            L2Model.fit(x=x, y=y, batch_size=args.batchSize, epochs=args.epochs, verbose=0)

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

    with open('results_online_test.txt', 'a') as outFile:
        outFile.write("Sequence_Id,Delay_To_be_Tested,Real_Output,Predicted_Output,Error_Count,Samples_Waiting_To_Be_Tested\n")
        while True:
            if len(testDataQueue) > 0:
                sequenceId = min(testDataQueue.keys()) # Get next sequence ID (min dictionary key)

                timestamp = testDataQueue[sequenceId]['timestamp']
                x = numpy.asarray(testDataQueue[sequenceId]['data']).astype(float).reshape((1, numberVariables))
                del testDataQueue[sequenceId]

                prediction = L1Model.predict(x, batch_size=1)
                mse = 0
                for j in range(len(prediction)):
                    mse += numpy.square(prediction[j]-x[0][j])
                mse /= len(prediction)

                prediction = LsModel.predict(mse, batch_size=1)
                if numpy.argmax(prediction[0]) > 0: # Was classified as "with activity", so need to go through second level to decide which is the activity
                    prediction = L2Model.predict(x, batch_size=1)

                currentTimestamp = datetime.datetime.now(timezone.utc).replace(tzinfo=timezone.utc).timestamp()

                if not sequenceId in predictions:
                    predictions[sequenceId] = {}
                predictions[sequenceId]['predicted'] = { 'timeDifference': (currentTimestamp-timestamp), 'output': prediction[0] }

                #print('Version', nnVersion)

            if len(predictions) > 0:
                sequenceId = min(predictions.keys()) # Get next sequence ID (min dictionary key)

                if 'predicted' in predictions[sequenceId] and 'real' in predictions[sequenceId]: # if already received expected output
                    timestampDifference = predictions[sequenceId]['predicted']['timeDifference']
                    predicted = numpy.argmax(predictions[sequenceId]['predicted']['output'])
                    real = numpy.argmax(numpy.asarray(predictions[sequenceId]['real']['output']).astype(float))

                    del predictions[sequenceId]

                    if predicted != real:
                        errorCount += 1

                    outFile.write(str(sequenceId)+","+str(timestampDifference)+","+str(real)+","+str(predicted)+","+str(errorCount)+","+str(len(predictions))+"\n")

                    #xResults.append(count)
                    #yResults.append(errorCount)
                    #count += 1

                    #plt.plot(xResults, yResults)
                    #plt.draw()
                    #plt.pause(0.001)
