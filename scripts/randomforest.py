#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#from sklearn.model_selection import train_test_split
#from scripts.multilabel_split import multilabel_train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score,
                             precision_score,
                             recall_score,
                            )



#Change to relative path later

print('loading train data')
x_train = np.load('data/Features/xception_train_all.npy').astype(np.float16) #np.load('data/Features/xception_train_all.npy')
print('loading labels')
y_train = np.load('data/Features/labels_train.npy')


#NEEDS TUNING! CANT RUN DUE TO OOM
params = {
 'n_estimators': 50, 
 'verbose': 2, 
 'n_jobs': 1,
}



# Train RF
print('Training RF and running predictions...')

rf = RandomForestClassifier(**params)
rf.fit(x_train[0:20000,], y_train[0:20000,])

print('loading validation data')
x_val = np.load('data/Features/xception_valid_all.npy')
y_val = np.load('data/Features/labels_validation.npy')

#Make predictions
preds = rf.predict(x_val)
print(preds)

#Turn into 0/1 encoding
preds = preds >0.5

#Print accuracy
acc = f1_score(y_val, preds,average='micro')
print('f1 score is' , acc)

