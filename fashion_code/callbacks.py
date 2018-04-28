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
                 num_thresholds=21, save_path=None):
        self.validation_generator = validation_generator
        self.test_generator = test_generator
        self.thresholds = np.linspace(0, 1, num_thresholds+2)[1:-1]
        self.save_fname = save_path
        self.save_path = join(paths['models'], save_path)

    def on_train_begin(self, logs={}):
        self.f1s = []
        self.precisions = []
        self.recalls = []

    def on_train_end(self, logs={}):
        if self.test_generator:
            print('Training done. Running predictions...')
            best_model = load_model(self.save_path)
            params = pd.read_csv(f'{self.save_path}-scores.csv')
            classes = pd.read_csv(paths['dummy']['csv']).columns
            threshold = params['threshold'].values[0]

            preds = best_model.predict_generator(self.test_generator,
                                                 use_multiprocessing=True,
                                                 workers=8,
                                                 verbose=1)
            preds = preds > threshold

            print('Converting labels...')
            mlb = MultiLabelBinarizer(classes=classes)
            mlb.fit(None) # necessary, won't actually do anything
            sparse_preds = mlb.inverse_transform(preds)

            submission_list = []
            for i, p in enumerate(sparse_preds, start=1):
                labels = ' '.join(p)
                submission_list.append([i, labels])

            submission_path = join(paths['results'],
                                   f'{self.save_fname}-submission.csv')
            print(f'Saving predictions to {submission_path}')
            columns = ['image_id', 'label_id']
            pd.DataFrame(submission_list, columns=columns) \
                        .to_csv(submission_path, index=False)

    def on_epoch_end(self, epoch, logs={}):
        print('Validating...')
        preds = self.model.predict_generator(self.validation_generator,
                                             use_multiprocessing=True,
                                             workers=8,
                                             verbose=1)
        preds = np.asarray(preds)
        targets = self.validation_generator.get_all_labels()

        local_f1s = []
        local_precs = []
        local_recs = []
        for th in self.thresholds:
            local_preds = preds > th
            f1 = f1_score(targets, local_preds, average='micro')
            prec = precision_score(targets, local_preds, average='micro')
            rec = recall_score(targets, local_preds, average='micro')

            local_f1s.append(f1)
            local_precs.append(prec)
            local_recs.append(rec)

        best_f1_idx = np.argmax(local_f1s)
        best_f1 = local_f1s[best_f1_idx]
        best_prec = local_precs[best_f1_idx]
        best_rec = local_recs[best_f1_idx]

        self.f1s.append(best_f1)
        self.precisions.append(best_prec)
        self.recalls.append(best_rec)

        logs['val_f1'] = best_f1

        print(f'f1: {best_f1:.4f} - precision: {best_prec:.4f} - '
              f'recall: {best_rec:.4f}')

        if best_f1 >= np.max(self.f1s) and self.save_path:
            print('F1 improved, saving model and scores...')
            self.model.save(self.save_path)
            scores = {
                'threshold': self.thresholds[best_f1_idx],
                'f1': best_f1,
                'precision': best_prec,
                'recall': best_rec,
            }
            df_path = join(f'{self.save_path}-scores.csv')
            pd.DataFrame(scores, index=[0]).to_csv(df_path)
