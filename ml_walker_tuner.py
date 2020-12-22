import uuid

import kerastuner as kt
import pandas as pd
import tensorflow as tf
from kerastuner import HyperParameters
from tensorflow.python.keras.layers import Dense, LeakyReLU, Dropout, ELU, PReLU
from tensorflow.python.keras.models import Sequential

import constants
import ml_splitter
import ml_transformer
from df_loader import load_ml_data
from xlogger import log

INPUT_DIMENSIONS = 352
DAYS_FOR_TEST = 10
MAX_TRIALS = 100


def build_model(hp: HyperParameters):
    # model hyper parameters
    initializer = hp.Choice('initializer',
                            ['random_normal', 'random_uniform', 'he_uniform', 'glorot_normal', 'glorot_uniform'])
    layers = hp.Int('layers', 1, 10, default=2)
    layer_size = hp.Choice('layer_size', [64, 256, 512, 1024])
    elu_alpha = hp.Float('elu_alpha', min_value=0.00001, max_value=1.0)
    lrelu_alpha = hp.Float('lrelu_alpha', min_value=0.00001, max_value=1.0)
    loss_function = hp.Choice('loss_function', ['mean_squared_error', 'huber_loss', 'poisson'])
    activation_function = hp.Choice('activation_function', ['sigmoid', 'relu'])
    network_type = hp.Choice('network_type', ['lrelu', 'prelu', 'relu', 'elu'])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate', 1e-7, 1e-2, sampling='log'),
        beta_1=hp.Float('beta1', min_value=0.5, max_value=0.9),
        beta_2=hp.Float('beta2', min_value=0.5, max_value=0.999999),
        epsilon=hp.Float('epsilon', min_value=1e-10, max_value=1e-01, sampling='log'),
        amsgrad=hp.Boolean('ams_grad'))
    dropout_on = hp.Boolean('dropout_on')
    dropout_rate = hp.Float('dropout_rate', 0.05, 0.95)

    model = Sequential()
    for i in range(0, layers):
        if i == 0:
            model.add(Dense(layer_size, kernel_initializer=initializer, input_dim=INPUT_DIMENSIONS))
        else:
            if network_type == 'elu':
                model.add(Dense(layer_size, kernel_initializer=initializer))
                model.add(ELU(alpha=elu_alpha))
            elif network_type == 'relu':
                model.add(Dense(layer_size, kernel_initializer=initializer, activation='relu'))
            elif network_type == 'prelu':
                model.add(Dense(layer_size, kernel_initializer=initializer))
                model.add(PReLU())
            elif network_type == 'lrelu':
                model.add(Dense(layer_size, kernel_initializer=initializer))
                model.add(LeakyReLU(alpha=lrelu_alpha))
            else:
                raise KeyError(network_type)
        if dropout_on is not None:
            model.add(Dropout(dropout_rate))
    # add the output
    model.add(Dense(1, kernel_initializer=initializer, activation=activation_function))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


class WalkerTuner(kt.Tuner):

    def run_trial(self, trial, **kwargs):

        train_and_validation = kwargs.get('train_and_validation')
        test = kwargs.get('test')
        test_x, test_y = test.iloc[:, 1:], test.iloc[:, :1]  # get the test set we'll use at the end

        # define the hyper parameters
        hp = trial.hyperparameters
        steps = hp.Int('training_steps', min_value=16, max_value=1024)
        validation_window_multiplier = hp.Int('validation_window_multiplier', min_value=1, max_value=16)
        step_batch_size = hp.Int('batch_size', min_value=16, max_value=1024)
        step_epochs = hp.Int('step_epochs', min_value=1, max_value=32)
        model = self.hypermodel.build(trial.hyperparameters)

        # `self.on_epoch_end` reports results to the `Oracle` and saves the
        # current state of the Model. The other hooks called here only log values
        # for display but can also be overridden. For use cases where there is no
        # natural concept of epoch, you do not have to call any of these hooks. In
        # this case you should instead call `self.oracle.update_trial` and
        # `self.oracle.save_model` manually.
        records = train_and_validation.shape[0]
        records_per_step = int(records / steps)
        # Step through the dataset
        for epoch in range(0, 1):
            for step in range(0, steps):
                self.on_batch_begin(trial, model, step, logs={})

                # get the ranges we'll work from
                val_start = records_per_step * (step + 1)
                val_end = val_start + (validation_window_multiplier * records_per_step)

                # make sure there is sufficient validation data available to run an iteration
                if (records - val_start) < records_per_step:
                    break

                # slice our training and validation sets
                train, val = train_and_validation[0:val_start], train_and_validation[val_start:val_end]

                # Get the train and val data, then train the model
                tx, ty = train.iloc[:, 1:], train.iloc[:, :1]
                vx, vy = val.iloc[:, 1:], val.iloc[:, :1]
                model.fit(tx, ty, validation_data=(vx, vy),
                          batch_size=step_batch_size,
                          epochs=step_epochs,
                          callbacks=None,
                          verbose=0)
                # evaluate per step
                loss_test = model.evaluate(test_x, test_y, verbose=0)
                self.on_batch_end(trial, model, step, logs={'root_mean_squared_error': loss_test[1]})
                log(f"{step}, {len(train)}, {len(val)}, {loss_test[0]}, {loss_test[1]}")
        # evaluate per tuner epic
        loss_test = model.evaluate(test_x, test_y, verbose=0)
        self.on_epoch_end(trial, model, 0, logs={'root_mean_squared_error': loss_test[1]})


def save(model, model_name: str):
    log("saving model")
    model.save(f"models/{model_name}", overwrite=True, include_optimizer=True, save_format='tf')


def get_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # load the data and perform the split
    train_and_validation, _, test = ml_splitter.split(load_ml_data(), 0, DAYS_FOR_TEST)
    train_and_validation = ml_transformer.transform(train_and_validation, for_prediction=False)
    test = ml_transformer.transform(test, for_prediction=False)
    return train_and_validation, test


def search(model_name):
    walker_tuner = WalkerTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('root_mean_squared_error', direction='min'),
            max_trials=MAX_TRIALS),
        hypermodel=build_model,
        directory='hypersearch',
        project_name='project01')

    train_and_validation, test = get_data()
    walker_tuner.search(train_and_validation=train_and_validation, test=test)

    best_hps = walker_tuner.get_best_hyperparameters()[0]
    print(best_hps.values)
    best_model = walker_tuner.get_best_models()[0]


# TODO: make getting the input dimensions dynamic!
pd.options.display.max_columns = 4
pd.options.display.max_rows = 1000
pd.options.display.max_info_columns = 1000
search(constants.PREDICTED_NEW_CASES)
