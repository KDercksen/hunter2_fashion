#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
from fashion_code.constants import paths
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.lib.io import file_io
import io
import numpy as np
import os
import pandas as pd


def read_img(fname, size, gcp=False):
    if gcp:
        with file_io.FileIO(fname, 'rb') as f:
            image_bytes = f.read()
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize(size, Image.BILINEAR)
            return img_to_array(img)
    else:
        img = load_img(fname, target_size=size)
        return img_to_array(img)


def create_submission(y_pred, filename):
        preds = y_pred > .5
        classes = pd.read_csv(paths['dummy']['csv'])

        print('Converting labels...')
        mlb = MultiLabelBinarizer(classes=classes)
        mlb.fit(None) # necessary, won't actually do anything
        sparse_preds = mlb.inverse_transform(preds)

        submission_list = []
        for i, p in enumerate(sparse_preds, start=1):
            labels = ' '.join(p)
            submission_list.append([i, labels])

        submission_path = join(paths['results'],
                               '{}-submission.csv'.format(filename))
        print('Saving predictions to {}'.format(submission_path))
        columns = ['image_id', 'label_id']
        pd.DataFrame(submission_list, columns=columns) \
                    .to_csv(submission_path, index=False)
