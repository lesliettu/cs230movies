"""
Random useful functions
"""

import os.path

def get_label_from_score(score):
	"""
	Given a tomato score between 0 and 100, 
	return its corresponding bucket
	"""
	if score <= 19: return 0
	if score <= 34: return 1
	if score <= 48: return 2
	if score <= 60: return 3
	if score <= 69: return 4
	if score <= 78: return 5
	if score <= 83: return 6
	if score <= 90: return 7
	if score <= 97: return 8
	if score <= 100: return 9

def get_score_from_path(path):
	basename = os.path.basename(path)
	return int(basename.split('_')[0])

def get_label_from_path(path):
	return get_label_from_score(get_score_from_path(path))	