# from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# first neural network with keras tutorial
# https://github.com/tensorflow/tensorflow/issues/18652
import torch
import torch.nn as nn
import covid_io
import covid_pandas

covid_pandas.configure()

# load the datasets
train = covid_io.read_prepared_dataset('train_20_data.csv.gz').sample(frac=1).reset_index(drop=True)
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, :1]
validation = covid_io.read_prepared_dataset('validation_20_data.csv.gz').sample(frac=1).reset_index(drop=True)
validation_x = validation.iloc[:, 1:]
validation_y = validation.iloc[:, :1]
test = covid_io.read_prepared_dataset('test_20_data.csv.gz').sample(frac=1).reset_index(drop=True)
test_x = test.iloc[:, 1:]
test_y = test.iloc[:, :1]

columns = train_x.shape[1]

