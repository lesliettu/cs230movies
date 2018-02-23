from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import util
import random
import collections
import math
import sys
import numpy as np
from sklearn.metrics import classification_report
import pickle

freq_threshold = 2

'''
	Input: 
		-list of examples with tuples in the form (bag of words, percent_fresh)
		-vocab that is None if this is train set otherwise use vocab returned from extractFeatures(train examples)
		-frequent_ngram_col_idx None for train set otherwise list of column indices corresponding to ngrams that pass the frequency threshold

	Vectorizes each example and transforms bag of words into frequencies of words in vocabulary
'''
def extractFeatures(examples, vocab=None, frequent_ngram_col_idx=None):
	corpus = [] # get bags of words for each trainingexample
	for x,y in examples:
		corpus.append(x)
	# corpus = np.array(examples[:,0]) 
	vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
	X = vectorizer.fit_transform(corpus)
	analyze = vectorizer.build_analyzer()
	fullfeature = X.toarray()
	print('SHAPE in Fit Model', len(fullfeature), len(fullfeature[0]))
	if not frequent_ngram_col_idx:
		sums = np.sum(fullfeature,axis=0)
		frequent_ngram_col_idx = np.nonzero([x > freq_threshold for x in sums])	# specify frequency threshold to include in vocab
	
	# consider passing in pruned vocab to not need next line for dev
	fullfeature = fullfeature[:,frequent_ngram_col_idx[0]]
	print('NEW SHAPE', len(fullfeature), len(fullfeature[0]))

	# TODO: append new features here especially separating out genre, rating

	return fullfeature, vectorizer.vocabulary_, frequent_ngram_col_idx

'''
	Extract ngram features from train set and use the returned vocabulary to extractFeatures
	ngram features on dev set

	Dump files as .pkl to be loaded in text_nn.py

	Run basic linear regression and output mean squared error
'''
def trainPredictor(trainExamples, devExamples):
	print('BEGIN: TRAIN')
	trainX, vocabulary, frequent_ngram_col_idx = extractFeatures(trainExamples)
	trainY = [y for x,y in trainExamples]
	trainX = np.array(trainX)
	trainY = np.reshape(np.array(trainY),(len(trainY),1))
	print('TRAIN X shape', trainX.shape)
	print('TRAIN Y shape', trainY.shape)
	pickle.dump(trainX, open('trainX.pkl', 'wb'))
	pickle.dump(trainY, open('trainY.pkl', 'wb'))

	# trainX = pickle.load(open('trainX.pkl', 'rb'))

	regr = LinearRegression()
	regr.fit(trainX, trainY)
	trainPredict = regr.predict(trainX)
	
	print("coefficient of acoustic", regr.coef_)
	print("TRAIN Mean squared error", mean_squared_error(trainY, trainPredict))
	print("TRAIN Variance score", r2_score(trainY, trainPredict))


	devX, _, _ = extractFeatures(devExamples, vocab=vocabulary, frequent_ngram_col_idx=frequent_ngram_col_idx)
	devY = [y for x,y in devExamples]

	devX = np.array(devX)
	devY = np.reshape(np.array(devY),(len(devY),1))
	print('DEV X shape', devX.shape)
	print('DEV Y shape', devY.shape)
	pickle.dump(devX, open('devX.pkl', 'wb'))
	pickle.dump(devY, open('devY.pkl', 'wb'))

	devPredict = regr.predict(devX)
	print("DEV Mean squared error", mean_squared_error(devY, devPredict))
	print("DEV Variance score", r2_score(devY, devPredict))

	
	return vocabulary, frequent_ngram_col_idx, regr



trainExamples = util.readExamples('moviesS.train')
devExamples = util.readExamples('moviesS.dev')
testExamples = util.readExamples('moviesS.test')

trainPredictor(trainExamples, devExamples)
