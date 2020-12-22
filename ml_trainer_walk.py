import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LeakyReLU, Dropout, ELU, PReLU
from tensorflow.python.keras.models import Sequential

import constants
import ml_splitter
import ml_transformer
from constants import PREDICTED_NEW_CASES
from df_loader import load_ml_data
from xlogger import log


class HP:
    KERNEL_INITIALIZER = 'random_normal'
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)
    METRICS = [tf.keras.metrics.RootMeanSquaredError()]
    LOSS = 'poisson'
    LAYER_SIZE = 200
    LAYERS = 3
    LAYER_DROPOUT = False
    LAYER_DROPOUT_RATE = 0.50
    OUTPUT_ACTIVATION = 'sigmoid'
    DAYS_FOR_VALIDATION = 0
    DAYS_FOR_TEST = 10
    TRAINING_EPOCHS = 2
    TRAINING_BATCH_SIZE = 32
    TRAINING_STEPS = 128
    ROUNDS = 1
    VERBOSE = 0
    EARLY_STOPPING_PATIENCE = 1000
    CALLBACKS = []  # EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)]


def get_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # load the data and perform the split
    train_and_validation, _, test = ml_splitter.split(load_ml_data(), 0, HP.DAYS_FOR_TEST)
    train_and_validation = ml_transformer.transform(train_and_validation, for_prediction=False)
    test = ml_transformer.transform(test, for_prediction=False)
    return train_and_validation, test


def get_model_sigmoid(model, dimensions) -> None:
    for i in range(0, HP.LAYERS):
        if i == 0:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, input_dim=dimensions))
        else:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, activation='sigmoid'))
        # add dropout
        if HP.LAYER_DROPOUT:
            model.add(Dropout(HP.LAYER_DROPOUT_RATE))


def get_model_tanh(model, dimensions) -> None:
    for i in range(0, HP.LAYERS):
        if i == 0:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, input_dim=dimensions))
        else:
            model.add(Dense(HP.LAYER_SIZE, kernel_initializer=HP.KERNEL_INITIALIZER, activation='tanh'))
        # add dropout
        if HP.LAYER_DROPOUT:
            model.add(Dropout(HP.LAYER_DROPOUT_RATE))


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
    test_x, test_y = test.iloc[:, 1:], test.iloc[:, :1]  # get the test set we'll use at the end
    model = get_model(train_and_validation.shape[1] - 1)
    records = train_and_validation.shape[0]
    records_per_step = int(records / HP.TRAINING_STEPS)

    # The number of training rounds
    for z in range(0, HP.ROUNDS):
        # Step through the dataset
        for step in range(0, HP.TRAINING_STEPS):
            val_start = records_per_step * (step + 1)
            val_end = val_start + records_per_step
            current_train, current_validation = (train_and_validation[0:val_start],
                                                 train_and_validation[val_start:val_end])

            # Get the train and val data, then train the model
            tx, ty = current_train.iloc[:, 1:], current_train.iloc[:, :1]
            vx, vy = current_validation.iloc[:, 1:], current_validation.iloc[:, :1]
            model.fit(tx, ty, validation_data=(vx, vy), batch_size=HP.TRAINING_BATCH_SIZE, epochs=HP.TRAINING_EPOCHS,
                      callbacks=HP.CALLBACKS, verbose=HP.VERBOSE)

            # Evaluate the test data
            loss_test = model.evaluate(test_x, test_y, verbose=HP.VERBOSE)
            log(f"R={z}[{step}/P{HP.TRAINING_STEPS}] "
                f"T={len(current_train)}, V={len(current_validation)}, SCORE = {loss_test}")

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
