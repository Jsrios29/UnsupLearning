# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:09:49 2020

This file takes the output of GLRM.py, and uses it to classify the validation data. The validation
rate is used to determine the performance of the model by computing the prediction rates.

the prediction rate is the total correct prediction over the number of predictions

@author: Juan Rios
"""
import numpy as np
import pandas as pd
#import GLRM as gl

# chunk width is 5000, or same as defined in GLRM.py
chunkWidth = 5000

# STEP 1: Load the X and Y matrices for each chunk of the training data 
# 0: columns 0 - 4999 of trainign data, 1: 5000 - 9999, 2: 10000 - 14999 and so on
#NOTE: this will only work for a chunkSize of 5000 in GLRM.py

X0 = pd.read_csv(r"outputX_0.csv")
X0 = X0.to_numpy()

X1 = pd.read_csv(r"outputX_1.csv")
X1 = X1.to_numpy()

X2 = pd.read_csv(r"outputX_2.csv")
X2 = X2.to_numpy()

X3 = pd.read_csv(r"outputX_3.csv")
X3 = X3.to_numpy()

X4 = pd.read_csv(r"outputX_4.csv")
X4 = X4.to_numpy()

X5 = pd.read_csv(r"outputX_5.csv")
X5 = X5.to_numpy()

Y0 = pd.read_csv(r"outputY_0.csv")
Y0 = Y0.to_numpy()

Y1 = pd.read_csv(r"outputY_1.csv")
Y1 = Y1.to_numpy()

Y2 = pd.read_csv(r"outputY_2.csv")
Y2 = Y2.to_numpy()

Y3 = pd.read_csv(r"outputY_3.csv")
Y3 = Y3.to_numpy()

Y4 = pd.read_csv(r"outputY_4.csv")
Y4 = Y4.to_numpy()

Y5 = pd.read_csv(r"outputY_5.csv")
Y5 = Y5.to_numpy()

valData = pd.read_csv(r"valid_task_1_2.csv")
valData = valData.to_numpy()

valHeight, valWidth = valData.shape

# STEP 2: for each entry in valData, compare the true
# value at row i and column j, versus the computed value at i and j of XY
# where the value is computed as xi dot yj. The false and true positive and negatives
# are counted. The prediction accuracy is outputted, along with a confusion matrix.
   
#True Positive, False Positive, True Negative, False Negative 
TP = 0 
FP = 0
TN = 0
FN = 0

counter = 0
for sample in valData:
    #find the appropriate matrix chunk, row and column, and the true label associated
    row = sample[1]
    col = sample[0]
    label = sample[3]
    chunkIdx = int(np.floor(col/chunkWidth))
    
    # find the prediction from the X and Y matrices
    if chunkIdx == 0:
        pred = np.dot(X0[row,:],Y0[:,col%chunkWidth])
    elif chunkIdx == 1:
        pred = np.dot(X1[row,:],Y1[:,col%chunkWidth])
    elif chunkIdx == 2:
        pred = np.dot(X2[row,:],Y2[:,col%chunkWidth])
    elif chunkIdx == 3:
        pred = np.dot(X3[row,:],Y3[:,col%chunkWidth])
    elif chunkIdx == 4:
        pred = np.dot(X4[row,:],Y4[:,col%chunkWidth])
    elif chunkIdx == 5:
        pred = np.dot(X5[row,:],Y5[:,col%chunkWidth])
    else:
        print(" An error accured, the chunkIdc did not fit any of the 6 matrices")
      
    # map the predictions, if less than 0, predict as 0, else as 1    
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

print("Final validation rate = " + str((TP + TN)/valHeight))
print("TN = " + str(TN))
print("TP = " + str(TP))
print("FN = " + str(FN))
print("FP = " + str(FP))






