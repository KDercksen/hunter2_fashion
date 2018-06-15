#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:56:45 2018

@author: stephan
@Subject: Train and predictions on validation set

@Approach:
    Given neural nets trained on the training sets we train the neural networks
    and test them on the validation set.

    We split the validation set into a holdout sample of 1000 images out of
    10000 and the reamining images will be used for training/generating samples

    Cross validation on the remaining validation set is performed as follow We
    train on N-1 of the N folds, and make predictions on the final fold.  This
    final fold will act as measure of fit (validation) and will be saved as a
    prediction. We repeat this process N times for all neural nets.

    The predictions on the folds will then be passed on to the second layer of
    the stack


@Comments:
    This file is to train the fine-tuned neural nets on the
    validation in a cross-validation manner.


"""

import sys
sys.path.insert(0, "../hunter2_fashion/scripts")
from fashion_code.callbacks import MultiGPUCheckpoint
from fashion_code.constants import paths
from fashion_code.generators import SequenceFromDisk
from keras.applications.xception import preprocess_input
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from model_builds import networks
from multiprocessing import Pool
from os.path import join
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa

# Define constants
num_epochs = 25
batch_size = 128


def load_network(net):
    model = load_model(join(paths['models'], '{}.h5'.format(net)))
    for l in model.layers[-25:]:
        l.trainable = True
    return model


def augment(batch):
    batch = preprocess_input(batch)
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


network = {'xfv3','xfv2'}

for net in network:
    print('Loading training data for {}...'.format(net))
    img_size = 200
    val_gen = SequenceFromDisk('validation', 64, (img_size, img_size))

    #Load the trained network from model_build
    #path_dict = paths
    #with tf.device('/cpu:0'):
    #    model = load_network(net)
    
    #model = multi_gpu_model(model)
#    opt = SGD(lr=1e-4, momentum=.9)
#    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    
    # cp = MultiGPUCheckpoint(join(paths['models'], '{}_val.h5'.format(net)),
                            # monitor='loss')

    # model.fit_generator(val_gen,
                        # epochs=num_epochs,
                        # use_multiprocessing=True,
                        # workers=8,
                        # callbacks=[cp])
    print('loading model')
    model = load_model(join(paths['models'], '{}_val'.format(net)))
    print('making preds')
    preds = model.predict_generator(val_gen, verbose=1)
    eval_score = f1_score(val_gen.get_all_labels(), preds > .2, average='micro')
    print('F1 score: {:.4f}'.format(eval_score))

    np.save(join(paths['models'], '{}_val_pred.npy'.format(net)), preds)

    clear_session()
