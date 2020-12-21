import keras
from kerastuner import HyperModel, BayesianOptimization

from kerastuner.tuners import RandomSearch
import sklearn
import tensorflow as tf

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dropout, ELU, Dense, PReLU, LeakyReLU


class MyHyperModel(HyperModel):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        # add the layers
        for i in range(hp.Int('num_layers', 2, 10)):
            model.add(
                layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))

        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='log_cosh',
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return model

    def add_layer(self, hp: HyperParameters,
                        model: keras.Sequential,
                        i: int,
                        input_dimensions) -> None:
        if i == 0:
            model.add(Dense(
                units=hp.Int(f"l{i}_units", min_value=32, max_value=512, step=32),
                kernel_initializer=hp.Choice(f"l{i}_init", ['random_normal', 'glorot_uniform']),
                input_dim=input_dimensions))
        else:
            model.add(Dense(units=hp.Int(f"l{i}_units", min_value=32, max_value=512, step=32),
                            kernel_initializer=hp.Choice(f"l{i}_init", ['random_normal', 'glorot_uniform'])))

            model.add(LeakyReLU(alpha=hp.Choice(f"l{i}_alpha", [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])))
            if hp.Boolean(f"l{i}_dropout"):
                model.add(Dropout(hp.Float(f"l{i}_dropout_rate", min_value=0.05, max_value=0.90)))
