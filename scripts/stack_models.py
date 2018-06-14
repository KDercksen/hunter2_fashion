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

from fashion_code.callbacks import MultiGPUCheckpoint
from fashion_code.constants import paths
from fashion_code.generators import SequenceFromDisk
from keras.applications.xception import preprocess_input
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from model_builds import networks
from multiprocessing import Pool
from os.path import join
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np

# Define constants
num_epochs = 25
batch_size = 128


def load_network(net):
    model = load_model(join(paths['models'], '{}.h5'.format(net)))
    return model


for net in networks.keys():
    print('Loading training data for {}...'.format(net))
    img_size = 299
    val_gen = SequenceFromDisk('validation', 128, (img_size, img_size), preprocess_input)

    #Load the trained network from model_build
    path_dict = paths
    model = load_network(net)
    opt = Adam(lr=5e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    cp = ModelCheckpoint(join(paths['models'], '{}_val.h5'.format(net)),
                         save_best_only=True, monitor='loss')

    model.fit_generator(val_gen,
                        epochs=num_epochs,
                        use_multiprocessing=True,
                        workers=8,
                        callbacks=[cp])

    model = load_model(join(paths['models'], '{}_val.h5'.format(net)))
    preds = model.predict_generator(val_gen, verbose=1)
    eval_score = f1_score(val_gen.get_all_labels(), preds > .2, average='micro')
    print('F1 score: {:.4f}'.format(eval_score))

    np.save(join(paths['models'], '{}_val_pred.npy'.format(net)), preds)

    clear_session()
