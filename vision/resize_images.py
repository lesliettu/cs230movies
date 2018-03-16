from PIL import Image
import os
from tqdm import tqdm
import json
#from pathlib import Path

SIZE = 64
OUTPUT_DIR = '/data/square/'

def reformat_filename(basename):
	parts = basename.split('_')
	return parts[0] + '_IMG_' + parts[1]

def resize_and_save(filename):
	image = Image.open(filename)
	image = image.resize((SIZE, SIZE), Image.BILINEAR)
	new_basename = reformat_filename(os.path.basename(filename))
	out_filename = os.path.join(os.getcwd()+OUTPUT_DIR, new_basename)
	#Path('out_filename').touch()
	image.save(out_filename)

failed = []

for subdir, dirs, files in os.walk(os.getcwd()+'/data/posters/'): # walk through data files
	for filename in files:
		#resize_and_save(os.path.join(subdir,filename))
		try:
			resize_and_save(os.path.join(subdir,filename))
		except:
			print('failed on', filename)
			failed.append(filename)

with open('resize_failed.json', 'w') as outfile:
	json.dumps(failed, outfile)