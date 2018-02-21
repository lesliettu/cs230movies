# remember to reactivate tensor flow when you open a new terminal window

import os
import pandas as pd
import numpy as np
import math
import keras
from keras.regularizers import l2
#Read in data
file_path = '/Users/petragrutzik/CSClasses/CS229/dota/dotaprediction/Input Data/'
file_name = 'dota2_pro_match_input_data_all.pkl'
df_all = pd.read_pickle(file_path + file_name)

CROSS_TRAIN = False
NUM_CHUNKS = 10

# #create train/dev set without cross train
# file_name = 'dota2_pro_match_input_data_train.pkl'
# df_train = pd.read_pickle(file_path + file_name)

# file_name = 'dota2_pro_match_input_data_dev.pkl'
# df_dev = pd.read_pickle(file_path + file_name)

# X_train = df_train.drop({'radiant_win', 'match_id'}, axis = 1)
# y_train = df_train['radiant_win']

# X_dev = df_dev.drop({'radiant_win', 'match_id'}, axis = 1)
# y_dev = df_dev['radiant_win']

#create train dev set for cross train

length_chunk = math.floor(len(df_all)/NUM_CHUNKS)

########################
#train a neural net
from keras.models import Sequential
from keras.layers import Dense, Activation

NUM_OUTPUT_UNITS = 1
print('numvars', np.shape(df_all)[1] - 3)
NUM_VARIABLES = np.shape(df_all)[1] - 3 # because to train, we will later drop 3 columns:'start_date','radiant_win', 'match_id'

model = Sequential()
model.add(Dense(200, input_dim=NUM_VARIABLES, activation='relu', W_regularizer=l2(0.5))) # 
model.add(keras.layers.core.Dropout(0.7))
model.add(Dense(200, activation='relu')) # ,W_regularizer=l2(0.5)
# model.add(keras.layers.core.Dropout(0.7))
# model.add(Dense(math.floor(NUM_VARIABLES/4), activation='relu'))
# model.add(Dense(math.floor(NUM_VARIABLES/6), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

df_dev = df_all[:length_chunk] 
df_train = df_all[length_chunk+1:] 

X_train = df_train.drop({'radiant_win', 'match_id', 'start_date'}, axis = 1)
y_train = df_train['radiant_win']

X_dev = df_dev.drop({'radiant_win', 'match_id', 'start_date'}, axis = 1)

y_dev = df_dev['radiant_win']
print(X_train)
print(y_train.values)
# Fit the model

model.fit(X_train.values, y_train.values, epochs=50, batch_size=1000)

# Test on dev
train_error = model.evaluate(X_train.values, y_train)
dev_error = model.evaluate(X_dev.values, y_dev)
print(dev_error)

dev_predictions = model.predict(X_dev.values)
dev_predictions = [round(x[0]) for x in dev_predictions]


file_path = '/Users/petragrutzik/CSClasses/CS229/dota/dotaprediction/Input Data/'
file_name = 'dota2_pro_match_input_data_test.pkl'
df_test = pd.read_pickle(file_path + file_name)

X_test = df_test.drop({'radiant_win', 'match_id', 'start_date'}, axis = 1)
y_test = df_test['radiant_win']

test_predictions = model.predict(X_test.values)
dev_error = model.evaluate(X_test.values, y_test)

test_output = pd.DataFrame({
         'prediction': list(test_predictions)
        ,'actual': list(y_test)
})
print(test_output)
test_output['correct'] = (test_output['prediction'] > 0.5) == test_output['actual']
test_correct_predictions = np.sum(test_output['correct'] == True)
test_size = np.shape(test_output)[0]
test_pct_correct = float(test_correct_predictions)/test_size

print("Test acc: " +str(test_pct_correct))