#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from fashion_code.constants import num_classes, paths, GCP_paths
from fashion_code.callbacks import MultiGPUCheckpoint
from fashion_code.generators import SequenceFromDisk, SequenceFromGCP
from imgaug import augmenters as iaa
from keras.applications import (xception,
                                inception_v3,
                                resnet50,
                                inception_resnet_v2,
                                densenet)
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from os.path import join
import sys
import keras.backend as K
import numpy as np


# Maps network names to module, constructor, input size
networks = {
    'xception': (xception, xception.Xception, 299),
    'resnet50': (resnet50, resnet50.ResNet50, 224),
    'inceptionv3': (inception_v3, inception_v3.InceptionV3, 299),
    'inceptionresnetv2': (inception_resnet_v2, inception_resnet_v2.InceptionResNetV2, 299),
    'densenet201': (densenet, densenet.DenseNet201, 224),
}


def build_model(NetworkConstr, num_classes):
    base_model = NetworkConstr(weights='imagenet', pooling='avg',
                               input_shape=(299, 299, 3), include_top=False)
    x = base_model.output
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(.5)(x)
    predictions = Dense(num_classes, activation='sigmoid',
                        kernel_initializer='he_normal')(x)
    model = Model(base_model.inputs, predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model


def train_model(NetworkConstr, aug_fun, preprocess_fun, args):
    # Some variables
    batch_size = args.batch_size
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
        model = build_model(NetworkConstr, num_classes)
        if args.multi_gpu:
            model = multi_gpu_model(model, cpu_relocation=True)
        model.compile(optimizer=optimizer, loss=loss)

    # Create data generators
    if args.gcp:
        train_gen = SequenceFromGCP('train', batch_size, img_size,
                                    preprocessfunc=preprocess_fun)
        valid_gen = SequenceFromGCP('validation', batch_size, img_size,
                                    preprocessfunc=preprocess_fun)
        if args.create_submission:
            test_gen = SequenceFromGCP('test', batch_size, img_size,
                                       preprocessfunc=preprocess_fun)
        else:
            test_gen = None
    else:
        train_gen = SequenceFromDisk('train', batch_size, img_size,
                                     preprocessfunc=aug_fun)
        valid_gen = SequenceFromDisk('validation', batch_size, img_size,
                                     preprocessfunc=preprocess_fun)
        if args.create_submission:
            test_gen = SequenceFromDisk('test', batch_size, img_size,
                                        preprocessfunc=preprocess_fun)
        else:
            test_gen = None

    # Fit model
    if args.multi_gpu:
        pm = MultiGPUCheckpoint(join(path_dict['models'], args.save_filename),
                                verbose=1)
    else:
        pm = ModelCheckpoint(join(path_dict['models'], args.save_filename),
                             monitor='val_loss',
                             save_best_only=True)

    train_steps = args.train_steps or len(train_gen)

    model.fit_generator(train_gen,
                        epochs=epochs,
                        steps_per_epoch=train_steps,
                        validation_data=valid_gen,
                        use_multiprocessing=use_multiprocessing,
                        workers=workers,
                        callbacks=[pm],
                        verbose=1)


if __name__ == '__main__':
    p = ArgumentParser('Simple stacking script.')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Batch size of images to read into memory')
    p.add_argument('--epochs', type=int, default=10,
                   help='Number of epochs to train')
    p.add_argument('--train-steps', type=int, help='Number of batches to '
                                                   'train on each epoch; if '
                                                   'not given, train on full '
                                                   'training set')
    p.add_argument('--save-filename', type=str,
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
    p.add_argument('--multi-gpu', action='store_true')
    args = p.parse_args()

    for net in networks.keys():
        NetworkConstr = networks['{}'.format(net)][1]
        preprocess_fun = networks['{}'.format(net)][0].preprocess_input

        def augment(batch):
            batch = preprocess_fun(batch)
            sometimes = lambda aug: iaa.Sometimes(.5, aug)
            seq = iaa.Sequential([
                iaa.Fliplr(.5),
                sometimes(iaa.Affine(rotate=(-20, 20))),
                sometimes(iaa.AddToHueAndSaturation((-20, 20))),
                sometimes(iaa.GaussianBlur((0, 2.))),
                sometimes(iaa.ContrastNormalization((.5, 1.5), per_channel=True)),
                sometimes(iaa.Sharpen(alpha=(0, 1.), lightness=(.75, 1.5))),
                sometimes(iaa.Emboss(alpha=(0, 1.), strength=(0, 2.))),
                sometimes(iaa.Crop(px=(5, 15))),
            ])
            return seq.augment_images(batch)

        args.save_filename = net + '.h5'
        train_model(NetworkConstr, augment, preprocess_fun, args)
        K.clear_session()

    sys.exit(0)
