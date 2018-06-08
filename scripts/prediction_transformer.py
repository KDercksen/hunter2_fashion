#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from fashion_code.constants import num_classes, paths
from fashion_code.generators import SequenceFromDisk
from fashion_code.util import create_submission
from keras.applications.xception import preprocess_input
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
import sys
import tensorflow as tf


def rf_transformer():
    return RandomForestClassifier(n_estimators=200, n_jobs=8, verbose=1)


def nn_transformer():
    inputs = Input(shape=(num_classes,))
    x = inputs
    for i in range(4):
        x = Dense(num_classes*(2+i), activation='relu',
                  name='trans_dense_{}'.format(i),
                  kernel_initializer='he_normal')(x)
        x = Dropout(.3, name='trans_dropout_{}'.format(i))(x)
    outputs = Dense(num_classes, activation='sigmoid', name='trans_out',
                    kernel_initializer='he_normal')(x)
    transformer = Model(inputs, outputs)

    optimizer = Adam(decay=1e-5)
    transformer.compile(optimizer=optimizer, loss='binary_crossentropy')
    return transformer


def generate_data(model, batch_size):
    train_gen = SequenceFromDisk('train', batch_size, (299, 299),
                                 preprocessfunc=preprocess_input)
    valid_gen = SequenceFromDisk('validation', batch_size, (299, 299),
                                 preprocessfunc=preprocess_input)

    x_train = model.predict_generator(train_gen,
                                      use_multiprocessing=True,
                                      workers=8,
                                      verbose=1)
    x_valid = model.predict_generator(valid_gen,
                                      use_multiprocessing=True,
                                      workers=8,
                                      verbose=1)
    y_train = np.load(join(paths['data'], 'labels_train.npy'))
    y_valid = np.load(join(paths['data'], 'labels_validation.npy'))

    return x_train, x_valid, y_train, y_valid


if __name__ == '__main__':
    p = ArgumentParser('Predict transformations for a given neural network')
    p.add_argument('filename', type=str,
                   help='The saved model to create initial predictions')
    p.add_argument('--save-filename', type=str,
                    help='Model to train transformer on top of')
    p.add_argument('--epochs', type=int, default=10, help='Epochs')
    p.add_argument('--batch-size', type=int, default=128, help='Batch size')
    p.add_argument('--create-submission', action='store_true')
    args = p.parse_args()

    batch_size = args.batch_size
    fname = join(paths['models'], args.filename)
    save_filename = args.save_filename
    model = multi_gpu_model(load_model(fname), cpu_relocation=True)
    threshold = .5
    epochs = args.epochs

    x_train, x_valid, y_train, y_valid = generate_data(model, batch_size)

    # TODO: add random forest classifier

    transformer = nn_transformer()
    transformer.fit(x_train, y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=epochs,
                    verbose=1)
    transformer.save(join(paths['models'], save_filename))

    valid_preds = transformer.predict(x_valid,
                                      batch_size=batch_size,
                                      verbose=1) > threshold
    score = f1_score(y_valid, valid_preds, average='micro')
    print('F1 score on validation: {:.4f}'.format(score))

    if args.create_submission:
        print('Creating submission...')
        test_gen = SequenceFromDisk('test', batch_size, (299, 299),
                                    preprocessfunc=preprocess_input)
        initial_preds = model.predict_generator(test_gen,
                                                use_multiprocessing=True,
                                                workers=8,
                                                verbose=1)
        y_pred = transformer.predict(initial_preds,
                                     batch_size=batch_size,
                                     verbose=1)
        create_submission(y_pred, save_filename)

    sys.exit(0)
