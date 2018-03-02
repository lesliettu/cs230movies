from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
#from nltk import pos_tag
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
import util
import random
import collections
import math
import sys
import numpy as np
from sklearn.metrics import classification_report
import pickle

freq_threshold = 2

def fitModel(examples, vocab=None, frequent_ngram_col_idx=None):
	corpus = [] # get bags of words for each trainingexample
	for x,y in examples:
		corpus.append(x)
	# corpus = np.array(examples[:,0]) 
	#print('corpus done')
	vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
	#print('vectorizer initialized')

	X = vectorizer.fit_transform(corpus)
	#print('fit transform thing done')

	#print 'X', X
	analyze = vectorizer.build_analyzer()
	#print('analyzer done')

	fullfeature = X.toarray()
	#fullfeature = X

	#print 'SHAPE in Fit Model', len(fullfeature), len(fullfeature[0])
	if not frequent_ngram_col_idx:
		sums = np.sum(fullfeature,axis=0)
		frequent_ngram_col_idx = np.nonzero([x > freq_threshold for x in sums])	# specify frequency threshold to include in vocab
	fullfeature = fullfeature[:,frequent_ngram_col_idx[0]]
	#print 'NEW SHAPE', len(fullfeature), len(fullfeature[0])

	# can append new features here

	return fullfeature, vectorizer.vocabulary_, frequent_ngram_col_idx

def trainPredictor(trainExamples):
	trainX, vocabulary, frequent_ngram_col_idx = fitModel(trainExamples)
	trainY = [y for x,y in trainExamples]
	trainY = np.reshape(trainY, (len(trainY),1))

	with open('trainX.pkl', 'w') as xtrain_out:
		pickle.dump(trainX,xtrain_out)

	with open('trainY.pkl', 'w') as ytrain_out:
		pickle.dump(trainY,ytrain_out)

	devX, _, __ = fitModel(devExamples,vocab=vocabulary,frequent_ngram_col_idx=frequent_ngram_col_idx)
	devY = [y for x,y in devExamples]
	devY = np.reshape(devY, (len(devY),1))

	print('shape of trainX', trainX.shape)
	print('shape of trainY', trainY.shape)
	print('shape of devX', devX.shape)
	print('shape of devY', devY.shape)


	with open('devX.pkl', 'w') as xdev_out:
		pickle.dump(devX,xdev_out)

	with open('devY.pkl', 'w') as ydev_out:
		pickle.dump(devY,ydev_out)




	"""
	regr = LinearRegression()
	regr.fit(trainX, trainY)
	trainPredict = regr.predict(trainX)
	print "hello"

	print "coefficient of acoustic", regr.coef_
	print "TRAIN Mean squared error", mean_squared_error(trainY, trainPredict)
	print "TRAIN Variance score", r2_score(trainY, trainPredict)


	devX, _, _ = fitModel(devExamples, vocab=vocabulary, frequent_ngram_col_idx=frequent_ngram_col_idx)
	devY = [y for x,y in devExamples]
	devPredict = regr.predict(devX)
	print "DEV Mean squared error", mean_squared_error(devY, devPredict)
	print "DEV Variance score", r2_score(devY, devPredict)

	"""
	#return vocabulary, frequent_ngram_col_idx, regr



trainExamples = util.readExamples('moviesS.train')
devExamples = util.readExamples('moviesS.dev')
testExamples = util.readExamples('moviesS.test')

trainPredictor(trainExamples)
