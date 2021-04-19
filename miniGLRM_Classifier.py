# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:09:55 2020

The purpose of this file is to evaluate the model trained on miniGLRM_collab.py. 
This model only trains on the first partition of the total training data, meaning
all the rows, columns 0 to 4999

@author: Juan Rios
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import random
import scipy.sparse
from scipy import sparse
import scipy.sparse


# load tbhe training data for the first 5000 features
Data = sparse.load_npz("first_5000_answers.npz")
# find the size of the data
numRows, numCols =  Data.shape

# find the colums means - the means of each colums
# find the variences of each colums
DataArray = Data.toarray()
colVars = np.var(DataArray, axis = 0)
colMeans = np.mean(Data, axis = 0)

X = pd.read_csv(r"outputX.csv")
X = X.to_numpy()

Y = pd.read_csv(r"outputY.csv")
Y = Y.to_numpy()

Z = np.dot(X,Y)
# z0 is the un-normalized result
Z0 = Z*colVars
Z = Z0 + colMeans

# Load the val data
valData = pd.read_csv(r"valid_task_1_2.csv")
valData = valData.to_numpy()

valHeight, valWidth = valData.shape

#True Positive, False Positive, True Negative, False Negative 
TP = 0 
FP = 0
TN = 0
FN = 0

counter = 0
for sample in valData:
    
    if ( sample[0] < 5000):
        
        pred = Z[sample[1],sample[0]]
        label = sample[3]
        
        # map the predictions
        if pred <= 0:
            pred = 0
        else:
            pred = 1
        
        # compute prediction rates
    
        if pred == 0 and label == 0:
            TN = TN + 1
        elif pred == 1 and label == 1:
            TP = TP + 1
        elif pred == 1 and label == 0:
            FP = FP + 1
        elif pred == 0 and label == 1:
            FN = FN + 1
        else:    
            print("Something went wrong with computing prediction rates, prediction and labels are : " + str(pred) + " " + str(label))
    

    # print out every 1000 rows:
    if (counter % 20000) == 0:
            print("Currently on " + str(counter) + " out of: " + str(valHeight))
        
    counter = counter + 1

print("Final validation rate = " + str((TP + TN)/((TN + TP + FN + FP))))
print("TN = " + str(TN))
print("TP = " + str(TP))
print("FN = " + str(FN))
print("FP = " + str(FP))