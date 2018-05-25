#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from fashion_code.constants import num_classes, paths
from fashion_code.generators import SequenceFromDisk
from keras.applications.xception import preprocess_input
from keras.layers import Dense, Input
from keras.models import Model, load_model
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import sys


def rf_transformer():
    model = RandomForestClassifier(n_estimators=50, n_jobs=8, verbose=1)
    return model


def nn_transformer():
    inputs = Input(shape=(num_classes,))
    x = Dense(num_classes*3, activation='relu',
              kernel_initializer='he_normal')(inputs)
    x = Dense(num_classes*2, activation='relu',
              kernel_initializer='he_normal')(x)
    outputs = Dense(num_classes, activation='sigmoid',
                    kernel_initializer='he_normal')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def generate_data(network):
    batch_size = 100
    train_steps = 200
    train_gen = SequenceFromDisk('train', batch_size, (299, 299),
                                 preprocess_input)
    valid_gen = SequenceFromDisk('validation', batch_size, (299, 299),
                                 preprocess_input)

    # Generate 20k training predictions to test
    x_train = network.predict_generator(train_gen, steps=train_steps,
                                        verbose=1)
    x_valid = network.predict_generator(valid_gen, verbose=1)

    y_train = np.load(paths['train']['labels'])[:batch_size*train_steps]
    y_valid = np.load(paths['validation']['labels'])

    return x_train, x_valid, y_train, y_valid


AVAILABLE_MODELS = {
    'rf': rf_transformer,
    'nn': nn_transformer,
}


if __name__ == '__main__':
    p = ArgumentParser('Predict transformations for a given neural network')
    p.add_argument('filename', type=str,
                   help='The saved model to create initial predictions')
    p.add_argument('transformer', type=str, choices=AVAILABLE_MODELS.keys(),
                   help='The type of model to train as prediction transformer')
    args = p.parse_args()

    fname = join(paths['models'], args.filename)
    network = load_model(fname)
    threshold = pd.read_csv(fname + '-scores.csv')['threshold'].values[0]
    transformer = AVAILABLE_MODELS[args.transformer]()

    x_train, x_valid, y_train, y_valid = generate_data(network)
    x_train = x_train > threshold
    x_valid = x_valid > threshold

    if args.transformer == 'nn':
        transformer.fit(x_train, y_train,
                        epochs=20,
                        verbose=1)
    else:
        # For sklearn transformers
        transformer.fit(x_train, y_train)
    preds = transformer.predict(x_valid) > threshold
    score = f1_score(y_valid, preds, average='micro')
    print(f'F1 score: {score}')

    sys.exit(0)
