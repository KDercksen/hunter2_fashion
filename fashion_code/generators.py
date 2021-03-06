#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .constants import paths, GCP_paths, num_classes
from .util import read_img
from keras.utils import Sequence
from tensorflow.python.lib.io import file_io
from os.path import join
import keras.backend as K
import numpy as np
import pandas as pd
import os


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

        print('SequenceFromDisk <{}>: {} samples'.format(mode, self.n_samples))

    def get_all_labels(self):
        if self.mode != 'test':
            return self.labels
        else:
            raise AttributeError(
                'This is a {} data generator, no labels available!'.format(self.mode))

    def __getitem__(self, idx):
        images = []

        start = idx * self.batch_size
        end = np.min([start + self.batch_size, self.n_samples])
        idxs = np.arange(start, end)

        for i in idxs:
            try:
                row = self.csv.iloc[i, :]
                img_id = row['imageId']
                img_path = join(self.paths['dir'], '{}.jpg'.format(img_id))
                img = read_img(img_path, self.img_size)
                images.append(img)
            except Exception as e:
                print('Failed to read index {}'.format(i))

        images = np.stack(images).astype(K.floatx())

        if self.preprocessfunc:
            images = self.preprocessfunc(images)

        if self.mode == 'test':
            return images
        else:
            return images, self.labels[idxs]

    def __len__(self):
        return self.n_batches


class SequenceFromGCP(Sequence):

    def __init__(self, mode, batch_size, img_size, preprocessfunc=None):
        self.mode = mode
        self.batch_size = batch_size
        self.img_size = img_size
        self.preprocessfunc = preprocessfunc
        self.data = []
        self.file_location = join(GCP_paths['data'], mode)
        print('Loading data from {}'.format(self.file_location))

        for fpath in file_io.get_matching_files(join(self.file_location, '*.jpg')):
            fname = fpath.split('/')[-1]
            if self.mode != 'test':
                split = fname[0:-4].split('_')
                labels = split[3][1:-1].split(',')
                labels = list(map(int, labels))
                self.data.append({'id': int(split[1]), 'labels': labels, 'file': fpath})
            else:
                self.data.append({'id': fname[0:-4], 'file': fpath})
        self.data = pd.DataFrame(self.data, columns=['id', 'labels', 'file'])
        self.n_samples = len(self.data)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

        print('SequenceFromGCP <{}>: {} samples'.format(self.mode, self.n_samples))

    def __getitem__(self, idx):
        images = []
        labels = []

        start = idx * self.batch_size
        end = np.min([start + self.batch_size, self.n_samples])
        idxs = np.arange(start, end)

        for i in idxs:
            try:
                row = self.data.iloc[i, :]
                img = read_img(row['file'], self.img_size, gcp=True)
                images.append(img)
                if self.mode != 'test':
                    label = row['labels']
                    label[:] = [i - 1 for i in label]
                    encoded = np.zeros(num_classes)
                    encoded[label] = 1
                    labels.append(encoded)
            except Exception as e:
                print('Failed to read index {}'.format(i))

        images = np.stack(images).astype(K.floatx())

        if self.preprocessfunc:
            images = self.preprocessfunc(images)

        if self.mode == 'test':
            return images
        else:
            return images, np.array(labels)

    def __len__(self):
        return self.n_batches
