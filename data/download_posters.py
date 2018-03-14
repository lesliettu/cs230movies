import urllib.request
import json
from PIL import Image
import os 
import sys

# Saved filenames will be TOMATOSCORE_EXAMPLEINDEX.jpg

IN_FILENAME = 'text_examples.json'
TEMP_FILENAME = 'temp.jpg'
SIZE_ORIG = (206, 305)
SIZE_SMALLER = (150, 222)
output_dir = os.getcwd() + '/posters/'
REPORT_INTERVAL = 1000

def resize_and_save(url, score, idx):
	try:
	    urllib.request.urlretrieve(url, TEMP_FILENAME)
	    image = Image.open(TEMP_FILENAME)
	    if image.size == SIZE_ORIG:
	    	out_filename = str(score) + '_' + str(idx) + '.jpg'
	    	image = image.resize(SIZE_SMALLER, Image.BILINEAR)
	    	image.save(os.path.join(output_dir, out_filename))
	except:
		print('Failed on index', idx, ', poster url', url)

if __name__ == '__main__':
	examples = []
	with open(IN_FILENAME, 'r') as infile:
		examples = json.load(infile)
	for idx in range(len(examples)):
		ex = examples[idx]
		resize_and_save(ex['poster'], ex['percent_fresh'], idx)
		if idx % REPORT_INTERVAL == 0:
			print('Saved example', idx)
