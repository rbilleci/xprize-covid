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
from tensorflow.python.keras.callbacks import EarlyStopping

import covid_constants

from pipeline import df_pipeline


class HP:
    KERNEL_INITIALIZER = 'glorot_uniform'
    OPTIMIZER = tf.keras.optimizers.Nadam()
    METRICS = [tf.keras.metrics.RootMeanSquaredError()]
    LOSS = tf.keras.losses.MeanSquaredError()
    LAYER_SIZE = 200  # 200
    LAYERS = 2  #
    LAYER_DROPOUT = False
    LAYER_DROPOUT_RATE = 0.25
    OUTPUT_ACTIVATION = 'sigmoid'  # sigmoid
    DAYS_FOR_VALIDATION = 28  # 24
    DAYS_FOR_TEST = 14  # 21
    TRAINING_EPOCHS = 20
    TRAINING_BATCH_SIZE = 32
    VERBOSE = 2
    EARLY_STOPPING_PATIENCE = 20
    CALLBACKS = [EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)]


def get_model_elu(model, dimensions) -> None:
    alpha = 0.1
    for i in range(0, HP.LAYERS):
        if i == 0:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, input_dim=dimensions))
        else:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER))
        model.add(ELU(alpha=alpha))
        # add dropout
        if HP.LAYER_DROPOUT:
            model.add(Dropout(HP.LAYER_DROPOUT_RATE))


def get_model_prelu(model, dimensions) -> None:
    for i in range(0, HP.LAYERS):
        if i == 0:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, input_dim=dimensions))
        else:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER))
        model.add(PReLU())
        # add dropout
        if HP.LAYER_DROPOUT:
            model.add(Dropout(HP.LAYER_DROPOUT_RATE))


def get_model_lrelu(model, dimensions) -> None:
    alpha = 0.1
    for i in range(0, HP.LAYERS):
        if i == 0:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, input_dim=dimensions))
        else:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER))
        model.add(LeakyReLU(alpha=alpha))
        # add dropout
        if HP.LAYER_DROPOUT:
            model.add(Dropout(HP.LAYER_DROPOUT_RATE))


def get_model(dimensions):
    model = Sequential()
    get_model_prelu(model, dimensions)
    model.add(Dense(1, kernel_initializer=HP.KERNEL_INITIALIZER, activation=HP.OUTPUT_ACTIVATION))
    model.compile(loss=HP.LOSS, optimizer=HP.OPTIMIZER, metrics=HP.METRICS)
    model.summary()
    return model


def get_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # train, validation, and test
    tr, v, test = df_pipeline.process_for_training(covid_constants.PATH_DATA_BASELINE,
                                                   HP.DAYS_FOR_VALIDATION,
                                                   HP.DAYS_FOR_TEST)
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
    # Train
    model = get_model(train_x.shape[1])
    history = model.fit(train_x,
                        train_y,
                        validation_data=(validation_x, validation_y),
                        batch_size=HP.TRAINING_BATCH_SIZE,
                        epochs=HP.TRAINING_EPOCHS,
                        callbacks=HP.CALLBACKS,
                        verbose=HP.VERBOSE)
    # Print data
    best_epoch = np.argmin(history.history['val_loss'])
    loss_train = history.history['loss'][best_epoch]
    loss_validation = history.history['val_loss'][best_epoch]
    loss_test = model.evaluate(test_x, test_y)
    print('loss, training:', loss_train)
    print('loss, validation:', loss_validation)
    print('loss, test:', loss_test)

    for i in range(0, 100):
        tx = train_x.iloc[i]
        ty = train_y.iloc[i]
        print(f"{model.predict(np.array([tx]))[0][0] * 1e6}\t\t{1e6 * ty['_label']}")

    save(model)


train()
