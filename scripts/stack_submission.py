# -*- coding: utf-8 -*-
"""
Subject: Sumission file for stacking models
"""

import sys
sys.path.append('../hunter2_fashion/scripts')
from os.path import join
import numpy as np
from tqdm import tqdm
from fashion_code.util import create_submission
from model_builds import networks
from fashion_code.constants import paths, GCP_paths
from fashion_code.generators import SequenceFromDisk, SequenceFromGCP
from keras.applications.xception import preprocess_input
from keras.models import load_model


'''
    TODO:
        1. paste/append the predictions of each individual model
            together and to a format the second CNN can read
        2. Make predictions for the second CNN file
        3. Save predictions into CSV file
        
        
'''

NUMFOLD = 5

# =============================================================================
# Load test data
# =============================================================================

val_gen = SequenceFromDisk('test',128, (200,200),preprocess_input)

#get all the data from the val sfd constructor
x_test = np.concatenate([val_gen[i][0] for i in tqdm(range(len(val_gen)))])


#
#One by one, load trained models and create predictions


# =============================================================================
# Load first stack models (CNN's)
# =============================================================================
#Loading xception,inception,resnet,densenet,inceptionresenet

'''
gcp IS false change later!!!
'''
gcp = False


for net in networks.keys():
    for fold in range(0,NUMFOLD):
        print('loading model {} for fold {}'.format(net,fold + 1))
        
        
        '''
        load model have incorrect folder specifications for gcp!!!
        REVIEW THIS
        '''
        
        
        #Load the trained network from model_build
        if gcp:
            path_dict = GCP_paths
        else:
            path_dict = paths
        
        #Load model and make predictions on test set
        model = load_model(join(paths['models'],'{}_fold_{}'.format(net,fold)))  
        preds = model.predict(x_test)
        
        ''' APPEND THE PREDICTIONS OF MODEL 1 TO THE NEXT'''
        
        '''APPENDING CODE HERE'''







filename = 'enterFILENAMEHERE'
secondCNNpred= 2 

create_submission(secondCNNpred, filename)
