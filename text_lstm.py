# ## 2 - Emojifier-V2: Using LSTMs in Keras: 
# 
# Let's build an LSTM model that takes as input word sequences. This model will be able to take word ordering into account. Emojifier-V2 will continue to use pre-trained word embeddings to represent words, but will feed them into an LSTM, whose job it is to predict the most appropriate emoji. 


import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from emo_utils import *
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras import optimizers
import json
from keras import backend as K

np.random.seed(1)
classes = 2


# GRAB DATA
X_train = read_file('data/binary_mov/train/sentences.txt')
Y_train = read_file('data/binary_mov/train/labels.txt')
X_dev = read_file('data/binary_mov/dev/sentences.txt')
Y_dev = read_file('data/binary_mov/dev/labels.txt')
X_test = read_file('data/binary_mov/test/sentences.txt')
Y_test = read_file('data/binary_mov/test/labels.txt')


maxLen_train = len(max(X_train, key=len).split())
maxLen_dev = len(max(X_dev, key=len).split())
maxLen_test = len(max(X_test, key=len).split())
maxLen = max(max(maxLen_train, maxLen_test), maxLen_dev)
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')



def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (1 line)
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split(' ')
        # Initialize j to 0
        j = 0
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            if w in word_to_index.keys():
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j = j+1
        
    return X_indices



def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape = input_shape, dtype= 'int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    print(embeddings.shape)
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.4)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=True)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.3)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    last_layer = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.3)(last_layer)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 10-dimensional vectors.
    X = Dense(classes, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    ### END CODE HERE ###
    
    return model


model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()
customAdam = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=customAdam, metrics=['accuracy'])

# It's time to train your model. Your Emojifier-V2 `model` takes as input an array of shape (`m`, `max_len`) and outputs probability vectors of shape (`m`, `number of classes`). We thus have to convert X_train (array of sentences as strings) to X_train_indices (array of sentences as list of word indices), and Y_train (labels as indices) to Y_train_oh (labels as one-hot vectors).

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
print(Y_train[:20])
Y_train_oh = convert_to_one_hot(Y_train, C = classes)    # C bins
history = model.fit(X_train_indices, Y_train_oh, validation_split=0.1111237, epochs = 17, batch_size = 512, shuffle=True)

inp = model.input                                           # input placeholder
outputs = [model.layers[-4].output]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function


'''
del model
model = load_model('nlp_experiments/keras_model/model_dev.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
# plot_model(model, to_file='model.png')
# for i in range(len(history.history['loss'])):
#     print('Epoch', i, 'train loss', history.history['loss'][i], 'train acc', history.history['acc'][i])
#     print('Epoch', i, 'dev loss', history.history['val_loss'][i], 'dev acc', history.history['val_acc'][i])

with open('train_pred.json', 'w') as outfile:
    json.dump(model.predict(X_train_indices).tolist(), outfile)

with open('train_layers.json', 'w') as outfile:
    layer_outs = functor([X_train_indices, 1.])
    json.dump(layer_outs[0].tolist(), outfile)
    print(layer_outs)

X_dev_indices = sentences_to_indices(X_dev, word_to_index, max_len = maxLen)

with open('dev_pred.json', 'w') as outfile:
    json.dump(model.predict(X_dev_indices).tolist(), outfile)

Y_dev_oh = convert_to_one_hot(Y_dev, C = classes)
loss, acc = model.evaluate(X_dev_indices, Y_dev_oh)
print()
print("Dev loss = ", loss, "accuracy = ", acc)

with open('dev_layers.json', 'w') as outfile:
    layer_outs = functor([X_dev_indices, 1.])
    json.dump(layer_outs[0].tolist(), outfile)
    print(layer_outs)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = classes)
X_test_predicted = model.predict(X_test_indices)

with open('test_pred.json', 'w') as outfile:
    json.dump(X_test_predicted.tolist(), outfile)

with open('test_layers.json', 'w') as outfile:
    layer_outs = functor([X_test_indices, 1.])
    json.dump(layer_outs[0].tolist(), outfile)
    print(layer_outs)

loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test loss = ", loss, "accuracy = ", acc)

plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('trainacc.png')
# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig('trainloss.png')


model.save('nlp_experiments/keras_model/model.h5')

