#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Some variables
    batch_size = 32
    epochs = 20
    loss = 'binary_crossentropy'
    optimizer = 'adam'
    validation_split = .1

    # Create synthetic training data
    x_train = np.random.random((100000, 100))
    y_train = x_train > .7

    # Define model
    inputs = Input(shape=x_train.shape[1:])
    x = Dense(400, activation='relu')(inputs)
    x = Dense(300, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    outputs = Dense(y_train.shape[1], activation='sigmoid')(x)
    model = Model(inputs, outputs)

    # Compile model
    model.compile(loss=loss, optimizer=optimizer)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_split=validation_split,
              epochs=epochs)

    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.show()
