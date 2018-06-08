#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:56:45 2018

@author: stephan
@Subject: Train and predictions on validation set

@Approach:
    Given trained, fine tuned neural nets from the training sets
    we train the neural networks and test them on the validation set.
    
    We split the validation set into a holdout sample of 1000 images out of 10000
    and the reamining images will be used for training/generating samples
    
    Cross validation on the remaining validation set is performed as follow
    We train on N-1 of the N folds, and make predictions on the final fold.
    This final fold will act as measure of fit (validation) and will be saved as
    a prediction. We repeat this process N times for all neural nets.
    
    The predictions on the folds will then be passed on to the second layer of the stack
    
    

@Comments:
    This file is to train the fine-tuned neural nets on the
    validation in a cross-validation manner. 
    

"""



import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # import KFold
from keras.applications.xception import preprocess_input
from model_builds import networks
from tqdm import tqdm
from fashion_code.generators import SequenceFromDisk, SequenceFromGCP

#First, split the validation set into two parts, a holdout and training
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1)

    




# Define constants
INPUT_SIZE = 299
n_pre_epochs = 10
n_epochs = 100
batch_size = 32
n_images = 100 



# CHANGE INPUT IMG SIZES LATER!
val_gen = SequenceFromDisk('validation',128, (200,200),preprocess_input)
batches = np.concatenate([val_gen[i][0] for i in tqdm(range(len(val_gen)))])



#This creates k fold validation indices
kf = KFold(n_splits=15, random_state=None, shuffle=False)
kf.get_n_splits(batches) # returns the number of splitting iterations in the cross-validator



i = 1
for train_index, test_index in kf.split(batches):
    
    print("TRAIN:", train_index, "TEST:", test_index)
    #X_train, X_test = batches[train_index], batches[test_index]
    i = i + 1
 #y_train, y_test = y[train_index], y[test_index]

