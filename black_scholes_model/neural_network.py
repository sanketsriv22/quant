import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *

import keras
import tensorflow as tf
from keras import Sequential
from keras import layers

from sklearn.model_selection import train_test_split

ticker = create_ticker('msft')  # this can be changed to user input later via GUI
options_data = gather_options_data(ticker)
X, y_true = features_and_label(options_data)  # DataFrames X, y_true
X, y_true = normalize_data(X, y_true)

model = Sequential()
model.add(layers.Dense(200, input_shape=(X.shape[1],)))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(1, activation='elu'))

model.compile(loss='mse', optimizer='adam')

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

predictions = model.predict(X_test)
plt.figure(figsize = (15,10))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.plot([0,3], [0,3], 'r')
plt.grid(True)
plt.show()


# try the model on apple's call options
ticker = create_ticker('aapl')  # this can be changed to user input later via GUI
options_data = gather_options_data(ticker)
X, y_true = features_and_label(options_data)  # DataFrames X, y_true
X, y_true = normalize_data(X, y_true)
predictions = model.predict(X)
plt.scatter(y_true, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.plot([0,3], [0,3], 'r')
plt.grid(True)
plt.show()