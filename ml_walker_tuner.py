import kerastuner as kt
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LeakyReLU, Dropout, ELU, PReLU
from tensorflow.python.keras.models import Sequential
from kerastuner import HyperParameters
import ml_splitter
import ml_transformer
from constants import PREDICTED_NEW_CASES
from df_loader import load_ml_data
from xlogger import log


def get_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # load the data and perform the split
    train, validation, test = ml_splitter.split(load_ml_data(), 31, 7)
    train = ml_transformer.transform(train, for_prediction=False)
    validation = ml_transformer.transform(validation, for_prediction=False)
    test = ml_transformer.transform(test, for_prediction=False)
    return train, validation, test


def build_model(hp: HyperParameters):
    # model hyper parameters
    initializer = hp.Choice('initializer',
                            ['random_normal', 'random_uniform', 'he_uniform', 'glorot_normal', 'glorot_uniform'])
    layers = hp.Int('layers', 2, 10)
    layer_size = hp.Choice('layer_size', [128, 256, 512, 1024, 2048])
    elu_alpha = hp.Float('elu_alpha', min_value=0.01, max_value=1.0)
    lrelu_alpha = hp.Float('lrelu_alpha', min_value=0.01, max_value=1.0)
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


def save(model, model_name: str):
    log("saving model")
    model.save(f"models/{model_name}", overwrite=True, include_optimizer=True, save_format='tf')


def walk_and_train(model_name: str):
    train, val, test = get_data()
    print(f"input dimensions should be{(train.shape[1] - 1)}")
    tuner = kt.Hyperband(build_model,
                         objective='val_root_mean_squared_error',
                         max_epochs=30,
                         hyperband_iterations=2)

    tx, ty = train.iloc[:, 1:], train.iloc[:, :1]
    vx, vy = val.iloc[:, 1:], val.iloc[:, :1]

    tuner.search(tx, ty,
                 validation_data=(vx, vy),
                 epochs=30,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])


# Train to get the model for confirmed cases
INPUT_DIMENSIONS = 352
pd.options.display.max_columns = 4
pd.options.display.max_rows = 1000
pd.options.display.max_info_columns = 1000

walk_and_train(PREDICTED_NEW_CASES)
