#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from fashion_code.callbacks import F1Utility
from fashion_code.constants import num_classes, paths
from fashion_code.generators import SequenceFromDisk
from keras.applications.xception import preprocess_input
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from os.path import join
import numpy as np
import sys


def nn_transformer(model):
    x = model.output
    for i in range(4):
        x = Dense(num_classes*(2+i), activation='relu',
                  name='trans_dense_{}'.format(i),
                  kernel_initializer='he_normal')(x)
    outputs = Dense(num_classes, activation='sigmoid', name='trans_out',
                    kernel_initializer='he_normal')(x)
    transformer = Model(model.inputs, outputs)

    for layer in model.layers:
        layer.trainable = False

    optimizer = Adam(decay=1e-5)
    transformer.compile(optimizer=optimizer, loss='binary_crossentropy')
    return transformer


if __name__ == '__main__':
    p = ArgumentParser('Predict transformations for a given neural network')
    p.add_argument('filename', type=str,
                   help='The saved model to create initial predictions')
    p.add_argument('--save-filename', type=str,
                    help='Model to train transformer on top of')
    p.add_argument('--epochs', type=int, default=10, help='Epochs')
    p.add_argument('--batch-size', type=int, default=128, help='Batch size')
    p.add_argument('--train-steps', type=int, help='Steps')
    p.add_argument('--create-submission', action='store_true')
    args = p.parse_args()

    batch_size = args.batch_size
    fname = join(paths['models'], args.filename)
    model = load_model(fname)
    threshold = .5
    epochs = args.epochs

    train_gen = SequenceFromDisk('train', batch_size, (299, 299),
                                 preprocessfunc=preprocess_input)
    valid_gen = SequenceFromDisk('validation', batch_size, (299, 299),
                                 preprocessfunc=preprocess_input)

    if args.create_submission:
        test_gen = SequenceFromDisk('test', batch_size, (299, 299),
                                    preprocessfunc=preprocess_input)
    else:
        test_gen = None

    train_steps = args.train_steps or len(train_gen)
    transformer = nn_transformer(model)

    pm = F1Utility(valid_gen, test_generator=test_gen,
                   save_path=paths['models'], save_fname=args.save_filename)

    transformer.fit_generator(train_gen,
                              epochs=epochs,
                              steps_per_epoch=train_steps,
                              use_multiprocessing=True,
                              workers=8,
                              callbacks=[pm],
                              verbose=1)

    sys.exit(0)
