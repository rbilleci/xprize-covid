import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense, LeakyReLU, Dropout, PReLU, ELU
from tensorflow.python.keras.models import Sequential
import tensorflow as tf
import numpy as np

import constants
import ml_splitter
import ml_transformer
from constants import PREDICTED_NEW_CASES
from df_loader import load_ml_data
from xlogger import log


class HP:
    KERNEL_INITIALIZER = 'random_normal'  # 'random_normal'
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)  # )
    METRICS = [tf.keras.metrics.RootMeanSquaredError()]
    LOSS = tf.keras.losses.MeanSquaredError()
    LAYER_SIZE = 200  # 200
    LAYERS = 2  # 2
    LAYER_DROPOUT = False
    LAYER_DROPOUT_RATE = 0.50
    OUTPUT_ACTIVATION = 'sigmoid'  # sigmoid
    DAYS_FOR_VALIDATION = 0  # 31
    DAYS_FOR_TEST = 10  # 14
    TRAINING_EPOCHS = 100
    TRAINING_BATCH_SIZE = 32
    VERBOSE = 2
    EARLY_STOPPING_PATIENCE = 1000
    CALLBACKS = []  # EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)]


def get_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # load the data and perform the split
    train_and_validation, _, test = ml_splitter.split(load_ml_data(), 0, HP.DAYS_FOR_TEST)
    train_and_validation = ml_transformer.transform(train_and_validation, for_prediction=False)
    test = ml_transformer.transform(test, for_prediction=False)
    return train_and_validation, test


def get_model_relu(model, dimensions) -> None:
    for i in range(0, HP.LAYERS):
        if i == 0:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, input_dim=dimensions))
        else:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, activation='relu'))
        # add dropout
        if HP.LAYER_DROPOUT:
            model.add(Dropout(HP.LAYER_DROPOUT_RATE))


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


def save(model, model_name: str):
    log("saving model")
    model.save(f"models/{model_name}", overwrite=True, include_optimizer=True, save_format='tf')


def walk_and_chew_gum(model_name: str):
    train_and_validation, test = get_data()
    test_x, test_y = test.iloc[:, 1:], test.iloc[:, :1]

    model = get_model(train_and_validation.shape[1] - 1)
    records = train_and_validation.shape[0]
    records_per_step = int(records / 100)
    epochs_per_step = 2

    i = 0
    while i < records:
        i += records_per_step
        current_train, current_validation = train_and_validation[0:i], train_and_validation[i:i + records_per_step]
        print('train=%d, validation=%d' % (len(current_train), len(current_validation)))
        tx, ty = current_train.iloc[:, 1:], current_train.iloc[:, :1]
        vx, vy = current_validation.iloc[:, 1:], current_validation.iloc[:, :1]
        model.fit(tx,
                  ty,
                  validation_data=(vx, vy),
                  batch_size=HP.TRAINING_BATCH_SIZE,
                  epochs=epochs_per_step,
                  callbacks=HP.CALLBACKS,
                  verbose=0)
        # model.train_on_batch()
        # model.reset_states()
        loss_test = model.evaluate(test_x, test_y)
        log(f"{loss_test}")

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

walk_and_chew_gum(PREDICTED_NEW_CASES)
