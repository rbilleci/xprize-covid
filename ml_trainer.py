import numpy as np
import pandas as pd
import tensorflow as tf

import constants
import ml_splitter
import ml_transformer

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from df_loader import load_ml_data
from constants import *
from xlogger import log


class HP:
    KERNEL_INITIALIZER = 'random_normal'  # 'random_normal'
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)#)
    METRICS = [tf.keras.metrics.RootMeanSquaredError()]
    LOSS = tf.keras.losses.MeanSquaredError()  # 'log_cosh'
    LAYER_SIZE = 200  # 200
    LAYERS = 2  # 2
    LAYER_DROPOUT = False
    LAYER_DROPOUT_RATE = 0.25
    OUTPUT_ACTIVATION = 'sigmoid'  # sigmoid
    DAYS_FOR_VALIDATION = 60  # 31
    DAYS_FOR_TEST = 10  # 14
    TRAINING_EPOCHS = 100
    TRAINING_BATCH_SIZE = 32
    VERBOSE = 2
    EARLY_STOPPING_PATIENCE = 100
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
    # load the data and perform the split
    df_train, df_val, df_test = ml_splitter.split(load_ml_data(), HP.DAYS_FOR_VALIDATION, HP.DAYS_FOR_TEST)
    # df_train, df_val, df_test = ml_splitter.split_random(load_ml_data())
    # df_train, df_val, df_test = ml_splitter.split_random_with_reserved_test(load_ml_data(), HP.DAYS_FOR_TEST)

    # transform the data for the neural network
    tr = ml_transformer.transform(df_train, for_prediction=False).sample(frac=1).reset_index(drop=True)
    val = ml_transformer.transform(df_val, for_prediction=False).sample(frac=1).reset_index(drop=True)
    test = ml_transformer.transform(df_test, for_prediction=False).sample(frac=1).reset_index(drop=True)

    return tr.iloc[:, 1:], tr.iloc[:, :1], val.iloc[:, 1:], val.iloc[:, :1], test.iloc[:, 1:], test.iloc[:, :1]


def save(model, model_name: str):
    log("saving model")
    model.save(f"models/{model_name}", overwrite=True, include_optimizer=True, save_format='tf')


def train(model_name: str):
    train_x, train_y, validation_x, validation_y, test_x, test_y = get_data()

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
    log(f"loss, training:{loss_train}")
    log(f"loss, validation: {loss_validation}")
    log(f"loss, test: {loss_test}")

    for i in range(0, 20):
        # expected value
        df_output = test_y.iloc[i]
        expected = df_output[PREDICTED_NEW_CASES]

        # predicted value
        df_input = test_x.iloc[i]
        predicted = model.predict(np.array([df_input]))[0][0]

        # perform scaling
        expected = expected * constants.INPUT_SCALE[PREDICTED_NEW_CASES]
        predicted = predicted * constants.INPUT_SCALE[PREDICTED_NEW_CASES]
        log(f"EXPECTED/PREDICTED:\t{expected} vs {predicted}")
    save(model, model_name)


# Train to get the model for confirmed cases

pd.options.display.max_columns = 4
pd.options.display.max_rows = 1000
pd.options.display.max_info_columns = 1000

train(PREDICTED_NEW_CASES)
