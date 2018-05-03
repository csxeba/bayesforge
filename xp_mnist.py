import numpy as np

from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist

from bayesify import Bayesify
from bdense import BayesianDense

(lX, lY), (tX, tY) = mnist.load_data()
lX, tX = lX / 255., tX / 255.
onehot = np.eye(10)
lY, tY = onehot[lY], onehot[tY]

inputs = Input(lX.shape[1:])
x = Bayesify(Dense(30, activation="tanh"))(inputs)
x = Dense(lY.shape[1], activation="softmax")(x)

ann = Model(inputs=inputs, outputs=x)
ann.compile(optimizer="adam", loss="categorical_crossentropy")
ann.fit(lX, lY, batch_size=64, epochs=10, validation_data=(tX, tY))
