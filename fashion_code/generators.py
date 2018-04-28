#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .constants import paths
from .util import read_img
from keras.utils import Sequence
from os.path import join
import keras.backend as K
import numpy as np
import pandas as pd


class SequenceFromDisk(Sequence):

    def __init__(self, mode, batch_size, img_size, preprocessfunc=None):
        self.mode = mode
        self.batch_size = batch_size
        self.img_size = img_size
        self.preprocessfunc = preprocessfunc
        self.paths = paths[mode]
        self.csv = pd.read_csv(self.paths['csv'])
        self.n_samples = len(self.csv)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

        if self.mode != 'test':
            self.labels = np.load(self.paths['labels'])

        print(f'SequenceFromDisk <{mode}>: {self.n_samples} samples')

    def get_all_labels(self):
        if self.mode != 'test':
            return self.labels
        else:
            raise AttributeError(f'This is a {self.mode} data generator, no '
                                 f'labels available!')

    def __getitem__(self, idx):
        images = []

        start = idx * self.batch_size
        end = np.min([start + self.batch_size, self.n_samples])
        idxs = np.arange(start, end)

        for i in idxs:
            try:
                row = self.csv.iloc[i, :]
                img_id = row['imageId']
                img_path = join(self.paths['dir'], f'{img_id}.jpg')
                img = read_img(img_path, self.img_size)
                images.append(img)
            except Exception as e:
                print(f'Failed to read index {i}')

        images = np.stack(images).astype(K.floatx())

        if self.preprocessfunc:
            images = self.preprocessfunc(images)

        if self.mode == 'test':
            return images
        else:
            return images, self.labels[idxs]

    def __len__(self):
        return self.n_batches
