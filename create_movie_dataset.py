

"""Read, split and save the movie json dataset for our model"""

import json
import os
import sys
import re

SMALL_SET = 1000
bins = [19, 34, 48, 60, 69, 78, 83, 90, 97, 100]


def load_dataset(path_csv):
    """Loads dataset into memory from csv file"""
    # Open the csv file, need to specify the encoding for python3
    use_python3 = sys.version_info[0] >= 3
    tags = []
    with (open(path_csv, encoding="windows-1252") if use_python3 else open(path_csv)) as f:
        dataset = []
        words, tag = [], []
        examples = json.load(f)
        # Each line of the csv corresponds to one word
        for example in examples:
            if 'percent_fresh' in example:
                tag = example['percent_fresh'] # bucket
                # bin_i = 0                       # map to correct bin
                # while (bin_i < len(bins)):
                #     if tag <= bins[bin_i]:
                #         tag = bin_i
                #         break
                #     bin_i+=1
                if tag <= 70:
                    tag = 0
                else:
                    tag = 1
                tags.append(tag)
                bag = ''
                for key in ['genre', 'title', 'description']:
                    if key in example and example[key]: 
                        bag += ' ' + example[key]
                words = re.findall(r"[\w']+|[.,!?;]",bag)
                dataset.append((words, tag))


    # chunk(sorted(tags), 10)

    return dataset

def chunk(xs, n):
    '''Split the list, xs, into n chunks'''
    L = len(xs)
    assert 0 < n <= L
    s, r = divmod(L, n)
    chunks = [xs[p:p+s] for p in range(0, L, s)]
    chunks[n-1:] = [xs[-r-s:]]
    for chunk in chunks:
        print(chunk[0], chunk[-1])

    return chunks

def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["good", "movie"], ["90"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for words, tags in dataset:
                file_sentences.write("{}\n".format(" ".join(words).lower()))
                file_labels.write("{}\n".format(tags))
    print("- done.")

def load_idx():
    train_idx = []
    dev_idx = []
    test_idx = []
    ### IMPORT
    return train_idx, dev_idx, test_idx


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/text_examples.json'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading Movie dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")

    # Split the dataset into train, dev and split (dummy split with no shuffle)
    print("Splitting according to movie indices...")

    train_idx, dev_idx, test_idx = load_idx()

    for i in train_idx:
        train_dataset.append(dataset[i])
    for i in dev_idx:
        dev_dataset.append(dataset[i])
    for i in test_idx:
        test_dataset.append(dataset[i])
        

    '''
    train_dataset = dataset[:int(0.85*len(dataset))]
    dev_dataset = dataset[int(0.7*len(dataset)) : int(0.85*len(dataset))]
    test_dataset = dataset[int(0.85*len(dataset)):]
    '''

    # Save the datasets to files
    save_dataset(train_dataset, 'data/binary/train')
    save_dataset(dev_dataset, 'data/binary/dev')
    save_dataset(test_dataset, 'data/binary/test')
