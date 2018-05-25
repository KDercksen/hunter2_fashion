#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
from keras.preprocessing.image import img_to_array, load_img
from multiprocessing import Pool
from sklearn.metrics import f1_score, roc_curve
import numpy as np


def read_img(fname, size):
    img = load_img(fname, target_size=size)
    return img_to_array(img)


def _single_threshold(y_true, y_pred, i):
    y = y_true[:, i]
    yp = y_pred[:, i]
    # If all values in y_true are negative, return 1 as threshold
    if not np.any(y):
        return 1
    _, _, thresholds = roc_curve(y, yp)
    results = []
    for th in thresholds[1:]:
        ypb = yp > th
        local_f1 = f1_score(y, ypb, average='micro')
        results.append((local_f1, th))
    best_th = max(results, key=lambda tup: tup[0])[1]
    return best_th


def find_thresholds(y_true, y_pred):
    thfunc = partial(_single_threshold, y_true, y_pred)
    with Pool() as p:
        final_thresholds = p.map(thfunc, range(y_true.shape[1]))

    return np.array(final_thresholds)
