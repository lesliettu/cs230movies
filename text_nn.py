# remember to reactivate tensor flow when you open a new terminal window
import os
import pandas as pd
import numpy as np
import math
import keras
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Activation

def train_run_nn():
        #Load data
        trainX = pickle.load(open('trainX.pkl', 'rb'))
        trainY = pickle.load(open('trainY.pkl', 'rb'))
        devX = pickle.load(open('devX.pkl', 'rb'))
        devY = pickle.load(open('devY.pkl', 'rb'))

        ########################
        #train a neural net
        NUM_OUTPUT_UNITS = 1

        NUM_VARIABLES = np.shape(trainX)[1] 

        model = Sequential()
        model.add(Dense(100, input_dim=NUM_VARIABLES, activation='relu')) #, W_regularizer=l2(0.5))) 
        #model.add(keras.layers.core.Dropout(0.7))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='relu'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(trainX.values, trainY.values, epochs=50, batch_size=None)

        # Test on dev
        train_error = model.evaluate(trainX.values, trainY)
        print(train_error)
        dev_error = model.evaluate(devX.values, devY)
        print(dev_error)

        dev_predictions = model.predict(devX.values)

train_run_nn()