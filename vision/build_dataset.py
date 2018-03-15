"""Split the poster dataset into train/dev/test.

Poster filenames are given by SCORE_INDEX.jpg, 
where the INDEX is the index position of the example in text_examples.json. 

Structure of the INPUT data-set:
    0_1532.jpg
    0_2423.jpg
    ...
    100_432.jpg

Structure of the OUTPUT data-set:
    train/
        0_1532.jpg
        ...
    dev/
        10_132.jpg
        ...
    test/
        32_34.jpg
        ...
"""

import argparse
import random
import os

from tqdm import tqdm
from shutil import copyfile

RANDOM_SEED = 230
TRAIN = .8
DEV = .1
# TEST = .1

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/posters', help="Directory with the posters dataset")
parser.add_argument('--output_dir', default='data/split_posters', help="Where to write the new data")

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith('.jpg')]

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(RANDOM_SEED)
    filenames.sort()
    random.shuffle(filenames)

    split_train = int(TRAIN * len(filenames))
    split_dev = int((TRAIN+DEV) * len(filenames))
    train_filenames = filenames[:split_train]
    dev_filenames = filenames[split_train:split_dev]
    test_filenames = filenames[split_dev:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, split)
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            copyfile(filename, os.path.join(output_dir_split, os.path.basename(filename)))

    print("Done building dataset")
