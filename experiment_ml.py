# from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# first neural network with keras tutorial
# https://github.com/tensorflow/tensorflow/issues/18652
from keras.models import Sequential
from keras.layers import GaussianDropout
from keras.layers import AlphaDropout, Dropout
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import PReLU
import tensorflow as tf
import numpy as np
import pandas as pd

import df_pipeline


class HP:
    kernel_initializer = 'random_normal'
    optimizer = tf.keras.optimizers.Nadam()
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
    loss = tf.keras.losses.MeanSquaredError()
    neurons = 1000
    neuron_layers = 2
    days_for_validation = 24
    days_for_test = 21


def reference_model_elu(dimensions) -> Sequential:
    model = Sequential()
    alpha = 0.1
    model.add(Dense(HP.neurons, input_dim=dimensions, kernel_initializer=HP.kernel_initializer))
    model.add(ELU(alpha=alpha))
    model.add(Dense(100, kernel_initializer=HP.kernel_initializer))
    model.add(ELU(alpha=alpha))
    model.add(Dense(10, kernel_initializer=HP.kernel_initializer))
    model.add(ELU(alpha=alpha))
    model.add(Dense(1, kernel_initializer=HP.kernel_initializer))
    model.add(ELU(alpha=alpha))
    model.compile(loss=HP.loss, optimizer=HP.optimizer, metrics=HP.metrics)
    print(model.summary())
    return model


def reference_model_prelu(dimensions) -> Sequential:
    model = Sequential()
    model.add(Dense(HP.neurons, input_dim=dimensions, kernel_initializer=HP.kernel_initializer))
    model.add(PReLU())
    model.add(Dense(100, kernel_initializer=HP.kernel_initializer))
    model.add(PReLU())
    model.add(Dense(10, kernel_initializer=HP.kernel_initializer))
    model.add(PReLU())
    model.add(Dense(1, kernel_initializer=HP.kernel_initializer))
    model.compile(loss=HP.loss, optimizer=HP.optimizer, metrics=HP.metrics)
    print(model.summary())
    return model


def reference_model_lrelu(dimensions) -> Sequential:
    model = Sequential()
    alpha = 0.1
    model.add(Dense(HP.neurons, input_dim=dimensions, kernel_initializer=HP.kernel_initializer))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(100, kernel_initializer=HP.kernel_initializer))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(10, kernel_initializer=HP.kernel_initializer))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(1, kernel_initializer=HP.kernel_initializer))
    model.compile(loss=HP.loss, optimizer=HP.optimizer, metrics=HP.metrics)
    print(model.summary())
    return model


def get_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # train, validation, and test
    tr, v, test = df_pipeline.process_for_training('OxCGRT_latest.csv', HP.days_for_validation, HP.days_for_test)
    tr = tr.sample(frac=1).reset_index(drop=True)
    v = v.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    return tr.iloc[:, 1:], tr.iloc[:, :1], v.iloc[:, 1:], v.iloc[:, :1], test.iloc[:, 1:], test.iloc[:, :1]


# Get the data
train_x, train_y, validation_x, validation_y, test_x, test_y = get_data()

# Get the model
model = reference_model_lrelu(train_x.shape[1])

# Train the model
print("fitting model")
model.fit(train_x,
          train_y,
          validation_data=(validation_x, validation_y),
          batch_size=16,
          epochs=10000,
          verbose=2)

print("predicting")
score = model.evaluate(test_x, test_y, verbose=2)

for i in range(0, 100):
    tx = train_x.iloc[i]
    ty = train_y.iloc[i]
    print(f"{model.predict(np.array([tx]))[0][0] * 1e6}\t\t{1e6 * ty['_label']}")
