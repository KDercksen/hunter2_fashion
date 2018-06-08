#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.lib.io import file_io
import os
import io
from PIL import Image
import numpy as np


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
