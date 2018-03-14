
import json
import os

out_filename = 'examples_filtered.json'

examples = None
with open(out_filename, 'r') as infile:
	examples = json.load(infile)

print(examples[0])


'''

examples = []

for subdir, dirs, files in os.walk(os.getcwd()+'/old/'): # walk through data files
	for filename in files:
		print(os.path.join(subdir,filename))
		filepath = os.path.join(subdir, filename)
		with open(filepath, 'r') as infile:
			try:
				batch = json.load(infile)
				examples += batch
			except:
				continue

examples_filtered = list(filter(lambda ex: 'percent_fresh' in ex, examples))

print(len(examples), len(examples_filtered))

with open(out_filename, 'w') as outfile:
	json.dump(examples_filtered, outfile)

'''