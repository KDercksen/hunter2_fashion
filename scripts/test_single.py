#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fashion_code.generators import SequenceFromDisk
from keras.applications.xception import preprocess_input
from keras.models import load_model
from sklearn.metrics import f1_score


model = load_model('models/xception.h5')
model.compile('adam', 'binary_crossentropy')
val_gen = SequenceFromDisk('validation', 128, (299, 299), preprocess_input)
labels = val_gen.get_all_labels()
preds = model.predict_generator(val_gen, verbose=1)

print(f1_score(labels, preds > .2, average='micro'))
