import csv
import os
import random
import re
import json

def buildTrainDevTestSet():
	trainFile = open('moviesS.train', 'w')
	devFile = open('moviesS.dev', 'w')
	testFile = open('moviesS.test', 'w')
	train_random = 0.8
	dev_random = 0.9
	test_random = 1

	for subdir, dirs, files in os.walk(os.getcwd()+'/data/'): # walk through data files
		for filename in files:
			print os.path.join(subdir,filename)
			filepath = os.path.join(subdir, filename)
			with open(filepath, 'rb') as infile:
				try:
					examples = json.load(infile)
				except:
					continue
				for example in examples:
					if 'percent_fresh' in example:
						y = str(example['percent_fresh']) + ' '
						x = ''
						for key in ['title', 'genre', 'description']:
							if key in example and example[key]: 
								x += ' ' + example[key]
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