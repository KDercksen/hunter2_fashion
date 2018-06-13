#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from fashion_code.callbacks import F1Utility, FinetuningXception
from fashion_code.constants import num_classes, paths, GCP_paths
from fashion_code.generators import SequenceFromDisk, SequenceFromGCP
from keras.applications.xception import Xception, preprocess_input
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from os.path import join
import sys


def build_model(num_classes):
    base_model = Xception(weights='imagenet', pooling='avg', include_top=False)
    x = base_model.output
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    predictions = Dense(num_classes, activation='sigmoid',
                        kernel_initializer='he_normal')(x)
    model = Model(base_model.inputs, predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model


def train_model(args):
    # Some variables
    batch_size = args.batch_size * args.gpus
    chpt = args.continue_from_chpt
    epochs = args.epochs
    img_size = (299, 299)
    loss = 'binary_crossentropy'
    optimizer = Adam(decay=1e-6)
    use_multiprocessing = not args.windows
    workers = 0 if args.windows else 8
    if args.gcp:
        path_dict = GCP_paths
    else:
        path_dict = paths

    # Create and compile model
    if chpt:
        model = load_model(join(path_dict['models'], chpt))
    else:
        model = build_model(num_classes)
        if args.gpus > 1:
            model = multi_gpu_model(model, args.gpus)
    model.compile(optimizer=optimizer, loss=loss)

    # Create data generators
    if args.gcp:
        train_gen = SequenceFromGCP('train', batch_size, img_size,
                                    preprocessfunc=preprocess_input)
        valid_gen = SequenceFromGCP('validation', batch_size, img_size,
                                    preprocessfunc=preprocess_input)
        if args.create_submission:
            test_gen = SequenceFromGCP('test', batch_size, img_size,
                                       preprocessfunc=preprocess_input)
        else:
            test_gen = None
    else:
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
    save_file = join(path_dict['models'], args.save_filename)
    mc = ModelCheckpoint(save_file, monitor='val_loss', save_best_only=True)

    uc = FinetuningXception()

    train_steps = args.train_steps or len(train_gen)

    model.fit_generator(train_gen,
                        epochs=epochs,
                        steps_per_epoch=train_steps,
                        validation_data=valid_gen,
                        use_multiprocessing=use_multiprocessing,
                        workers=workers,
                        # This callback does validation, checkpointing and
                        # submission creation
                        callbacks=[mc,uc],
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
    p.add_argument('--gcp', action='store_true',
                   help='Change file loading for Google Cloud Platform')
    p.add_argument('--job-dir', type=str, help='Location of the job directory '
                                               'for the current GCP job')
    p.add_argument('--gpus', type=int, default=1,
                   help='Number of GPUs used for training')
    args = p.parse_args()

    train_model(args)

    sys.exit(0)
