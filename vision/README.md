# Predicting Tomato Score by Poster

*Authors: Leslie Tu, Petra Grutzik, and Kate Park*

## Data

The posters' filenames are given as SCORE_INDEX.jpg, where the INDEX is the index position of the example in text_examples.json.

We have 19,579 posters in total, with an 80/10/10 split for train/dev/test.

Prior to running build_dataset.py, the data directory should look like this:
```
data/
	posters/
		0_12.jpg
		0_414.jpg
		...
```