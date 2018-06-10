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


#Add model build file explicitly to the path
import sys
sys.path.append('../hunter2_fashion/scripts')


from os.path import join
import numpy as np
import random 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # import KFold
from keras.applications.xception import preprocess_input
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from model_builds import networks
from keras.layers import Dense
from keras.backend import clear_session
from fashion_code.constants import paths, GCP_paths
from fashion_code.generators import SequenceFromDisk, SequenceFromGCP
from sklearn.metrics import f1_score
from tqdm import tqdm

random.seed(123)

# Define constants
INPUT_SIZE = 299
n_pre_epochs = 10
num_epochs = 100
batch_size = 32
n_images = 100

#Number of cross validation folds 
NUMFOLD =5

# CHANGE INPUT IMG SIZES LATER!
# Load data from sequencefromDiks 
#ADD SEQUENCEFROMGCP AS WELL LATER

'''
IMG SIZE INPUT CHANGE
'''

val_gen = SequenceFromDisk('validation',128, (200,200),preprocess_input)

#get all the data from the val sfd constructor
x_train = np.concatenate([val_gen[i][0] for i in tqdm(range(len(val_gen)))])
y_train = val_gen.get_all_labels()

#First, split the validation set into two parts, a holdout and training
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)


#On the remainder of the training set, perform cross validation


#This creates k fold validation indices
kf = KFold(n_splits=NUMFOLD, random_state=None, shuffle=False)
# returns the number of splitting iterations in the cross-validator
kf.get_n_splits(x_train) 


#Train models using the created indicies



def load_network(net):
    
    model = load_model(join(paths['models'],'{}'.format(net)))

    #Load model and unfreeze only last 2 top layers
    for layer in (model.layers[:-2]):
        layer.trainable = False

    return model 



i = 1

'''
gcp IS false change later!!!
'''
gcp = False
for train_index, test_index in kf.split(x_train):
    print('Preparing indices/ cross validation data')
    x_fold_train, x_fold_test = x_train[train_index], x_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]
    for net in networks.keys():
        print('Loading training data for {}...'.format(net))
        
        
        '''
        load model have incorrect folder specifications for gcp!!!
        REVIEW THIS
        '''
        #Load the trained network from model_build
        if gcp:
            path_dict = GCP_paths
        else:
            path_dict = paths
            
        model = load_network(net)
         
        '''
        CODE FOR ACTUALLY FITTING THE NETWORK
        '''
        cp = ModelCheckpoint(join(paths['models'],'{}_fold_{}'.format(net,i)),
                             monitor = "val_loss",
                             save_best_only = True)
        
        model.fit(x_fold_train, y_fold_train, 
                  batch_size=batch_size,
                  epochs = num_epochs,
                  validation_data = (x_fold_test,y_fold_test),
                  callbacks = [cp])

        '''
        f1 SCORE VALIDATION HAS TO BE DONE
        '''
         
        '''
        predict labels and save best models
        '''
        #load model best obv cp
        model = load_model(join(paths['models'],'{}_fold_{}'.format(net,i)))     
        preds = model.predict(x_fold_test)
        #Evaluate score 
        eval_score = f1_score(y_fold_test, preds, average='micro')
        #Save predictions, along with the ordered labels for training
        np.save(join(paths['models'],'{}_fold_{}_pred_{}'.format(net,i,eval_score)),
                preds)
        clear_session()
        
    '''
    Save labels of the fold, for training next CNN
    '''
    
    #Save labels
    np.save(join(paths['models'],'ylabel_fold_{}_pred_{}'.format(i,eval_score)),
            y_fold_test)
    np.save(join(paths['models'],'trainind_fold_{}'.format(i)),
            train_index)
    np.save(join(paths['models'],'testind_fold_{}'.format(i)),
            test_index)    
    np.save(join(paths['models'],'holdout_data_x'),
             x_test) 
    np.save(join(paths['models'],'holdout_data_y'),
             y_test) 
        
    i = i + 1
        