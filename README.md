# cs230movies

add datafiles in data/ folder. We have a mini dataset in there for you!

In order to split dataset into train/dev/test formatted x y:
```
python create_datasets.py
```
In order to create .pkl files for neural network and run linear regression baseline:
```
python text_model.py
```
In order to run neural network on .pkl files outputted from text_model.py:
```
python text_nn.py
```