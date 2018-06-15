#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fashion_code.constants import num_classes
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Reshape
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np


def build_model(num_models):
    # Model takes (9897, 228, num_models)
    inputs = Input(shape=(num_classes, num_models))
    x = Reshape((num_classes, num_models, 1))(inputs)
    x = Conv2D(32, (1, 3))(x)
    x = Conv2D(64, (1, 3))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(.3)(x)
    outputs = Dense(num_classes, activation='sigmoid',
                    kernel_initializer='he_normal')(x)
    model = Model(inputs, outputs)
    return model


def load_predictions(*args):
    # args contains filenames of NumPy arrays with predictions
    return np.stack([np.load(fname) for fname in args], axis=-1)


if __name__ == '__main__':
    filenames = [
        'models/xception_val_pred.npy',
        'models/inceptionresnetv2_val_pred.npy',
        'models/resnet50_val_pred.npy',
        'models/densenet201_val_pred.npy',
        'models/inceptionv3_val_pred.npy',
    ]
    preds = load_predictions(*filenames)
    labels = np.load('data/labels_validation.npy')

    kf = KFold(5, shuffle=True, random_state=123)
    for i, (train_idx, val_idx) in enumerate(kf.split(preds), start=1):
        model = build_model(preds.shape[-1])
        opt = Adam(decay=1e-7)
        model.compile('adam', 'binary_crossentropy')

        cp = ModelCheckpoint('models/cnn_transformer_fold_{}.h5'.format(i),
                             monitor='val_loss',
                             save_best_only=True, verbose=1)

        model.fit(preds[train_idx], labels[train_idx],
                  batch_size=256,
                  epochs=100,
                  validation_data=(preds[val_idx], labels[val_idx]),
                  callbacks=[cp],
                  verbose=1)
