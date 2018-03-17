import csv
import os
import random
import re
import json

'''
	Process files in data folder to split into train/dev/test sets
	Percent_fresh is the score we attempt to predict, split this out as the y

	Treats title, genre and description as 1 bag of words
	TODO: move this into text_model.py to extract separate attributes
'''
def buildTrainDevTestSet():
	trainFile = open('moviesS_b.train', 'w')
	devFile = open('moviesS_b.dev', 'w')
	testFile = open('moviesS_b.test', 'w')
	train_random = 0.8
	dev_random = 0.9
	test_random = 1

	# for subdir, dirs, files in os.walk(os.getcwd()+'/data/'): # walk through data files
	# 	for filename in files:
	# 		print(os.path.join(subdir,filename))
	# 		filepath = os.path.join(subdir, filename)
	with open('data/text_examples.json', 'rb') as f:
		reader = json.loads(f)
		for example in reader:
			if example['percent_fresh']:
				if example['percent_fresh'] <= 70:
					y = '0 + '
				else:
					y = '1 + '
				# y = example['percent_fresh'] + ' '
				x = example['title'] + ' ' + example['genre'] + ' ' + example['description']	# treat all as bag of words-TODO: separate features
				x = " ".join(re.findall(r"[\w']+|[.,!?;]",x)).lower()	# split out punctuation
				line = y + ' ' + x +'\n'
				r = random.random()
				if r < train_random:
					trainFile.write(line)
				elif r < dev_random:
					devFile.write(line)
				else:
					testFile.write(line)


buildTrainDevTestSet()