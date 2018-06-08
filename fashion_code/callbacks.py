#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fashion_code.constants import paths
from keras.callbacks import Callback
from keras.models import load_model
from os.path import join
from sklearn.metrics import (f1_score,
                             precision_score,
                             recall_score,
                            )
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd


class F1Utility(Callback):

    def __init__(self, validation_generator, test_generator=None,
                 save_fname=None, save_path=None):
        super().__init__()
        self.validation_generator = validation_generator
        self.test_generator = test_generator
        self.save_fname = save_fname
        self.save_path = join(save_path, save_fname)

    def on_train_begin(self, logs={}):
        self.f1s = []
        self.precisions = []
        self.recalls = []

    def on_train_end(self, logs={}):
        if self.test_generator:
            print('Training done. Running predictions...')
            best_model = load_model(self.save_path)
            classes = pd.read_csv(paths['dummy']['csv']).columns

            preds = best_model.predict_generator(self.test_generator,
                                                 use_multiprocessing=True,
                                                 workers=8,
                                                 verbose=1)
            preds = preds > .5

            print('Converting labels...')
            mlb = MultiLabelBinarizer(classes=classes)
            mlb.fit(None) # necessary, won't actually do anything
            sparse_preds = mlb.inverse_transform(preds)

            submission_list = []
            for i, p in enumerate(sparse_preds, start=1):
                labels = ' '.join(p)
                submission_list.append([i, labels])

            submission_path = join(paths['results'],
                                   '{}-submission.csv'.format(self.save_fname))
            print('Saving predictions to {}'.format(submission_path))
            columns = ['image_id', 'label_id']
            pd.DataFrame(submission_list, columns=columns) \
                        .to_csv(submission_path, index=False)

    def on_epoch_end(self, epoch, logs={}):
        print('Validating...')
        preds = self.model.predict_generator(self.validation_generator,
                                             use_multiprocessing=True,
                                             workers=8,
                                             verbose=1)
        preds = preds > .5
        targets = self.validation_generator.get_all_labels()

        f1 = f1_score(targets, preds, average='micro')
        precision = precision_score(targets, preds, average='micro')
        recall = recall_score(targets, preds, average='micro')

        self.f1s.append(f1)
        self.precisions.append(precision)
        self.recalls.append(recall)

        logs['val_f1'] = f1

        print('f1: {:.4f} - precision: {:.4f} - recall: {:.4f}'
              .format(f1, precision, recall))

        if f1 >= np.max(self.f1s) and self.save_path:
            print('F1 improved, saving model and scores...')
            self.model.save(self.save_path)
            scores = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }
            df_path = join('{}-scores.csv'.format(self.save_path))
            pd.DataFrame(scores, index=[0]).to_csv(df_path)


class FinetuningXception(Callback):

    def __init__(self, validation_generator):
        super().__init__()

    def on_train_begin(self, logs={}):
        self.block_indices = []
        for i, layer in enumerate(self.model.layers):
            if layer.name[0:3] == 'add':
                self.block_indices.append(i)
        self.block_indices = self.block_indices[::-1]
        self.current_index = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.current_index < len(self.block_indices):
            print('Unfreezing block {}...'.format(self.current_index))
            for layer in self.model.layers[self.block_indices[self.current_index]:]:
                layer.trainable = True
            self.current_index += 1

