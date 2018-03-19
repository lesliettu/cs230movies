import json
import util
import numpy as np

with open('../data/text_examples.json', 'r') as f:
	text_examples = json.load(f)


def get_accuracy(text_weight):
	acc = {}
	predictions = {}
	for split in ['train', 'dev', 'test']:
		predictions[split] = {}
		with open('kate/'+split+'_indices.json', 'r') as f:
			indices = json.load(f)
		with open('kate/'+split+'_pred.json', 'r') as f:
			pred = json.load(f)

		for i, idx in enumerate(indices):
			predictions[split][idx] = {}
			predictions[split][idx]['prob_text'] = pred[i]
			predictions[split][idx]['pred_text'] = np.argmax(pred[i])
			predictions[split][idx]['actual'] = util.get_label_from_score(text_examples[idx]['percent_fresh'])

		with open('cnn/' + split + '_order.json', 'r') as f:
			indices_img = json.load(f)
			indices_img = [int(fn.split('/')[-1].split('.')[0].split('_')[2]) for fn in indices_img]

		with open('cnn/' + split + '_softmax.json', 'r') as f:
			pred_img = json.load(f)

		for i, idx in enumerate(indices_img):
			predictions[split][idx]['prob_img'] = pred_img[i]
			predictions[split][idx]['pred_img'] = np.argmax(pred_img[i])

		count_correct = 0.
		count_total = 0
		for idx in predictions[split]:
			text = np.array(predictions[split][idx]['prob_text'])
			img = np.array(predictions[split][idx]['prob_img'])
			avg = text*text_weight + img*(1-text_weight)
			predictions[split][idx]['prob_avg'] = avg.tolist()
			predictions[split][idx]['pred_avg'] = np.argmax(avg)
			count_total += 1
			if predictions[split][idx]['pred_avg'] == predictions[split][idx]['actual']:
				count_correct += 1

		acc[split] = count_correct/count_total
	return acc

for i in range(11):
	text_weight = i/10
	print('Text weight:', text_weight)
	acc = get_accuracy(text_weight)
	print('\t Train:', acc['train'], 'Dev:', acc['dev'], 'Test:', acc['test'])

