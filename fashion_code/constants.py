#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join

# Paths relative to the project root directory
data_dir = './data'
model_dir = './models'
result_dir = './results'
paths = {
    'data': data_dir,
    'models': model_dir,
    'results': result_dir,
    'train': {
        'dir': join(data_dir, 'train'),
        'labels': join(data_dir, 'labels_train.npy'),
        'csv': join(data_dir, 'labels_train.csv'),
    },
    'validation': {
        'dir': join(data_dir, 'validation'),
        'labels': join(data_dir, 'labels_validation.npy'),
        'csv': join(data_dir, 'labels_validation.csv'),
    },
    'test': {
        'dir': join(data_dir, 'test'),
        'csv': join(data_dir, 'test.csv'),
    },
    'dummy': {
        'csv': join(data_dir, 'dummy_label_col.csv'),
    },
}

# Number of classes
with open(paths['dummy']['csv']) as f:
    num_classes = len(f.read().strip().split(','))
