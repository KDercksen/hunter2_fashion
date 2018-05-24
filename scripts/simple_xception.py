#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from fashion_code.callbacks import F1Utility
from fashion_code.constants import num_classes, paths
from fashion_code.generators import SequenceFromDisk
from keras.applications.xception import Xception, preprocess_input
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense
from keras.models import Model, load_model
from os.path import join
import sys


def build_model(num_classes, freeze_base=True):
    base_model = Xception(weights='imagenet', pooling='avg',
                             include_top=False)
    x = base_model.output
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    predictions = Dense(num_classes, activation='sigmoid',
                        kernel_initializer='he_normal')(x)
    model = Model(base_model.inputs, predictions)

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    return model


def train_model(args):
    # Some variables
    batch_size = args.batch_size
    chpt = args.continue_from_chpt
    epochs = args.epochs
    img_size = (299, 299)
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'
    use_multiprocessing = True
    workers = 8

    # Create and compile model
    if chpt:
        model = load_model(join(paths['models'], chpt))
    else:
        model = build_model(num_classes)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    # Create data generators
    train_gen = SequenceFromDisk('train', batch_size, img_size,
                                 preprocessfunc=preprocess_input)
    valid_gen = SequenceFromDisk('validation', batch_size, img_size,
                                 preprocessfunc=preprocess_input)
    if args.create_submission:
        test_gen = SequenceFromDisk('test', batch_size, img_size,
                                    preprocessfunc=preprocess_input)
    else:
        test_gen = None

    # Fit model
    pm = F1Utility(valid_gen, test_generator=test_gen,
                   save_path=args.save_filename)
    lr = ReduceLROnPlateau(monitor='val_f1', patience=4, factor=.5)

    train_steps = args.train_steps or len(train_gen)

    if args.windows:
        use_multiprocessing = False
        workers = 0

    model.fit_generator(train_gen,
                        epochs=epochs,
                        steps_per_epoch=train_steps,
                        use_multiprocessing=use_multiprocessing,
                        workers=workers,
                        # This callback does validation, checkpointing and
                        # submission creation
                        callbacks=[pm, lr],
                        verbose=1)


if __name__ == '__main__':
    p = ArgumentParser('Simple Xception script.')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Batch size of images to read into memory')
    p.add_argument('--epochs', type=int, default=10,
                   help='Number of epochs to train')
    p.add_argument('--train-steps', type=int, help='Number of batches to '
                                                   'train on each epoch; if '
                                                   'not given, train on full '
                                                   'training set')
    p.add_argument('--save-filename', type=str, required=True,
                   help='Filename for saved model (will be appended to models '
                        'directory)')
    p.add_argument('--continue-from-chpt', type=str,
                   help='Continue training from h5 checkpoint file')
    p.add_argument('--create-submission', action='store_true',
                   help='If given, create a submission after training')
    p.add_argument('--windows', action='store_true',
                   help='Disable multiprocessing in order to run on Windows')
    args = p.parse_args()

    train_model(args)

    sys.exit(0)
