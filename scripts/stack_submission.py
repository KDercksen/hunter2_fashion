#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fashion_code.constants import paths
from fashion_code.generators import SequenceFromDisk
from fashion_code.util import create_submission
from keras.applications.xception import preprocess_input
from keras.models import load_model
from keras.utils import multi_gpu_model
from sklearn.metrics import f1_score
import keras.backend as K
import numpy as np
import tensorflow as tf


net_filenames = [
    'models/densenet201_val.h5',
    'models/inceptionresnetv2_val.h5',
    'models/inceptionv3_val.h5',
    'models/resnet50_val.h5',
    'models/xception_val.h5',
]

print('Creating test generator...')
test_gen = SequenceFromDisk('test', 128, (299, 299), preprocess_input)

print('Running predictions...')
preds = []
for net in net_filenames:
    with tf.device('/cpu:0'):
        model = load_model(net, compile=False)
    model = multi_gpu_model(model)
    p = model.predict_generator(test_gen,
                                use_multiprocessing=True,
                                workers=8,
                                verbose=1)
    preds.append(p)
    K.clear_session()

preds = np.stack(preds, axis=-1)
print(preds.shape)
np.save('results/stack_preds.npy', preds)

print('Transforming predictions...')
cnn_preds = []
for i in range(1, 6):
    cnn_filename = 'models/cnn_transformer_fold_{}.h5'.format(i)
    cnn_model = load_model(cnn_filename)
    cnn_p = cnn_model.predict(preds, batch_size=64, verbose=1)
    cnn_preds.append(cnn_p)

cnn_preds = np.stack(cnn_preds, axis=-1)
cnn_preds = np.mean(cnn_preds, axis=-1)
print('Done... Shape: {}'.format(cnn_preds.shape))

print('Creating submission...')
create_submission(preds, .15, 'cnn_stack')
print('Done!')
