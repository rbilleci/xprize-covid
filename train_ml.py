# from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# first neural network with keras tutorial
# https://github.com/tensorflow/tensorflow/issues/18652
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import PReLU
import tensorflow as tf
import numpy as np
import pandas as pd
import covid_constants

from pipeline import df_pipeline


class HP:
    kernel_initializer = 'glorot_uniform'
    optimizer = tf.keras.optimizers.Nadam()
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
    loss = tf.keras.losses.MeanSquaredError()
    hidden_layer_size = 200  # 200
    hidden_layer_count = 2  #
    hidden_layer_dropout = False
    hidden_layer_dropout_rate = 0.25
    output_layer_activation = 'sigmoid'  # sigmode
    days_for_validation = 28  # 24
    days_for_test = 14  # 21
    training_epochs = 20
    training_batch_size = 32
    verbose = 2


def get_model_elu(model, dimensions) -> None:
    alpha = 0.1
    for i in range(0, HP.hidden_layer_count):
        if i == 0:
            model.add(Dense(HP.hidden_layer_size, kernel_initializer=HP.kernel_initializer, input_dim=dimensions))
        else:
            model.add(Dense(HP.hidden_layer_size, kernel_initializer=HP.kernel_initializer))
        model.add(ELU(alpha=alpha))
        # add dropout
        if HP.hidden_layer_dropout:
            model.add(Dropout(HP.hidden_layer_dropout_rate))


def get_model_prelu(model, dimensions) -> None:
    for i in range(0, HP.hidden_layer_count):
        if i == 0:
            model.add(Dense(HP.hidden_layer_size, kernel_initializer=HP.kernel_initializer, input_dim=dimensions))
        else:
            model.add(Dense(HP.hidden_layer_size, kernel_initializer=HP.kernel_initializer))
        model.add(PReLU())
        # add dropout
        if HP.hidden_layer_dropout:
            model.add(Dropout(HP.hidden_layer_dropout_rate))


def get_model_lrelu(model, dimensions) -> None:
    alpha = 0.1
    for i in range(0, HP.hidden_layer_count):
        if i == 0:
            model.add(Dense(HP.hidden_layer_size, kernel_initializer=HP.kernel_initializer, input_dim=dimensions))
        else:
            model.add(Dense(HP.hidden_layer_size, kernel_initializer=HP.kernel_initializer))
        model.add(LeakyReLU(alpha=alpha))
        # add dropout
        if HP.hidden_layer_dropout:
            model.add(Dropout(HP.hidden_layer_dropout_rate))


def get_model(dimensions):
    model = Sequential()
    get_model_prelu(model, dimensions)
    model.add(Dense(1, kernel_initializer=HP.kernel_initializer, activation=HP.output_layer_activation))
    model.compile(loss=HP.loss, optimizer=HP.optimizer, metrics=HP.metrics)
    model.summary()
    return model


def get_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # train, validation, and test
    tr, v, test = df_pipeline.process_for_training(covid_constants.path_data_baseline,
                                                   HP.days_for_validation,
                                                   HP.days_for_test)
    tr = tr.sample(frac=1).reset_index(drop=True)
    v = v.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    return tr.iloc[:, 1:], tr.iloc[:, :1], v.iloc[:, 1:], v.iloc[:, :1], test.iloc[:, 1:], test.iloc[:, :1]


def save(model):
    print("saving model")
    model.save("model", overwrite=True, include_optimizer=True, save_format='tf')


def train():
    # Get the data
    train_x, train_y, validation_x, validation_y, test_x, test_y = get_data()

    # Get the models
    model = get_model(train_x.shape[1])

    # Train the models
    print("fitting models")
    model.fit(train_x,
              train_y,
              validation_data=(validation_x, validation_y),
              batch_size=HP.training_batch_size,
              epochs=HP.training_epochs,
              verbose=2)

    print("predicting")
    print(model.evaluate(test_x, test_y, verbose=HP.verbose))

    for i in range(0, 100):
        tx = train_x.iloc[i]
        ty = train_y.iloc[i]
        print(f"{model.predict(np.array([tx]))[0][0] * 1e6}\t\t{1e6 * ty['_label']}")

    save(model)


train()
