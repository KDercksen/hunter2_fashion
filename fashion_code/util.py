#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img


def read_img(fname, size):
    img = load_img(fname, target_size=size)
    return img_to_array(img)
