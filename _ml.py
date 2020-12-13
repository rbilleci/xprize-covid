# from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# first neural network with keras tutorial
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import covid_io
import covid_pandas

covid_pandas.configure()

# load the datasets
train = covid_io.read_prepared_dataset('train_20_data.csv.gz').sample(frac=1).reset_index(drop=True)
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, :1]
validation = covid_io.read_prepared_dataset('validation_20_data.csv.gz').sample(frac=1).reset_index(drop=True)
validation_x = validation.iloc[:, 1:]
validation_y = validation.iloc[:, :1]
test = covid_io.read_prepared_dataset('test_20_data.csv.gz').sample(frac=1).reset_index(drop=True)
test_x = test.iloc[:, 1:]
test_y = test.iloc[:, :1]

columns = train_x.shape[1]

model = Sequential()
model.add(Dense(1024, input_dim=columns, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="tanh"))

# TODO: review: https://www.pluralsight.com/guides/regression-keras


print("compiling model")
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer="adam",
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
print(model.summary())

print("fitting model")
model.fit(train_x,
          train_y,
          validation_data=(validation_x, validation_y),
          batch_size=128,
          epochs=10000,
          verbose=2)

print("predicting")
score = model.evaluate(test_x, test_y, verbose=2)
print(score)
