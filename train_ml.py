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

import datasets_constants
from oxford_constants import LABEL, CASES, PREDICTED_NEW_CASES

from pipeline import df_pipeline


class HP:
    KERNEL_INITIALIZER = 'glorot_uniform'
    OPTIMIZER = tf.keras.optimizers.Adam()
    METRICS = [tf.keras.metrics.RootMeanSquaredError()]
    LOSS = tf.keras.losses.MeanSquaredError()
    LAYER_SIZE = 200  # 200
    LAYERS = 2  # 2
    LAYER_DROPOUT = False
    LAYER_DROPOUT_RATE = 0.1
    OUTPUT_ACTIVATION = 'sigmoid'  # sigmoid
    DAYS_FOR_VALIDATION = 31  # 31
    DAYS_FOR_TEST = 14  # 14
    TRAINING_EPOCHS = 10
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


def get_data(column_to_predict: str) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    tr, v, test = df_pipeline.get_datasets_for_training(datasets_constants.PATH_DATA_BASELINE,
                                                        HP.DAYS_FOR_VALIDATION,
                                                        HP.DAYS_FOR_TEST,
                                                        column_to_predict)
    tr = tr.sample(frac=1).reset_index(drop=True)
    v = v.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    return tr.iloc[:, 1:], tr.iloc[:, :1], v.iloc[:, 1:], v.iloc[:, :1], test.iloc[:, 1:], test.iloc[:, :1]


def save(model, column_to_predict: str):
    print("saving model")
    model.save(f"models/{column_to_predict}", overwrite=True, include_optimizer=True, save_format='tf')


def train(column_to_predict: str, model_name: str):
    # Get the data
    train_x, train_y, validation_x, validation_y, test_x, test_y = get_data(column_to_predict)

    pd.options.display.max_columns = 4
    pd.options.display.max_rows = 100
    pd.options.display.max_info_columns = 1000
    train_x.info()
    print(train_x.sample(10))
    train_y.info()
    print(train_y.sample(10))

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
        print(f"{model.predict(np.array([tx]))[0][0] * 1e6}\t\t{1e6 * ty[LABEL]}")

    save(model, model_name)


# Train to get the model for confirmed cases
train(CASES, PREDICTED_NEW_CASES)
