import os, random, operator, sys
from collections import Counter
import sys

def readExamples(path):
    '''
    Reads a set of training examples.
    '''
    examples = []
    for line in open(path, 'rb'):
        # Format of each line: <percent_fresh> <bag of words>
        y, x = line.split(' ', 1)	# 1 split so splits out the score
        x = unicode(x.strip(), errors='ignore')
        examples.append((x, int(y)))

    print('Read %d examples from %s' % (len(examples), path))
    return examples