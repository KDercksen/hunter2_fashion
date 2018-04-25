#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.callbacks import Callback
from sklearn.metrics import (f1_score,
                             precision_score,
                             recall_score,
                            )
import numpy as np


class PrintMetrics(Callback):

    def __init__(self, num_thresholds=1):
        self.thresholds = np.linspace(0, 1, num=num_thresholds+2)[1:-1]

    def on_train_begin(self, logs={}):
        self.f1s = []
        self.recalls = []
        self.precisions = []

    def on_epoch_end(self, epoch, logs={}):
        preds = self.model.predict(self.validation_data[0])
        preds = np.asarray(preds).round()
        targets = self.validation_data[1]

        f1 = f1_score(targets, preds, average='micro')
        recall = recall_score(targets, preds, average='micro')
        precision = precision_score(targets, preds, average='micro')

        self.f1s.append(f1)
        self.recalls.append(recall)
        self.precisions.append(precision)

        print(f'f1: {f1:.4f} - precision: {precision:.4f} - '
              f'recall: {recall:.4f}')
