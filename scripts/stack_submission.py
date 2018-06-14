# -*- coding: utf-8 -*-
"""
Subject: Sumission file for stacking models
"""

from fashion_code.constants import paths
from fashion_code.generators import SequenceFromDisk
from fashion_code.util import create_submission
from keras.applications.xception import preprocess_input
from keras.models import load_model
import keras.backend as K
import numpy as np


net_filenames = [
    'models/densenet201_val.h5',
    'models/inceptionresnetv2_val.h5',
    'models/inceptionv3_val.h5',
    'models/resnet50_val.h5',
    'models/xception_val.h5',
]

cnn_filename = 'models/cnn_transformer.h5'

print('Creating test generator...')
test_gen = SequenceFromDisk('test', 128, (299, 299), preprocess_input)

print('Running predictions...')
preds = []
for net in net_filenames:
    model = load_model(net)
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
cnn_model = load_model(cnn_filename)
preds = cnn_model.predict(preds, batch_size=64, verbose=1)

print('Creating submission...')
create_submission(preds, 'cnn_stack')
print('Done!')
