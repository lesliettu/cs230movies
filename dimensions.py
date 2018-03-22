import json

with open('test_layers.json', 'r') as f:
	layer = json.load(f)
	print(len(layer), len(layer[0]))

with open('dev_layers.json', 'r') as f:
	layer = json.load(f)
	print(len(layer), len(layer[0]))

with open('train_layers.json', 'r') as f:
	layer = json.load(f)
	print(len(layer), len(layer[0]))