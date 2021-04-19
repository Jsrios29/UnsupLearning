# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:46:20 2020

General Low Rank Model:
    
The purpose of this script is to run our sparse matrix containing the 
training data through this GLR Model. BY running this model, we will 
be able to reconstruct a low-rank approximation to our data matrix, 
and thus allow us to predict missing values.

For a tutorial on methods:
https://docs.h2o.ai/h2o-tutorials/latest-stable/tutorials/glrm/glrm-tutorial.html  something about huge dataset
https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.ipynb
https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/glrm/glrm.walking.gait.py <--- good tutorial

NOTE: This code uses the complete training data, however, since the competition has ended, no submissions are allowed
thus in order to test this model, use the local_data_split.py code provided by the competition, which partitions the training data into
a training set and val set. Train on this training set and test on the validation set.

@author: Juan Rios
"""

import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2OGeneralizedLowRankEstimator
from scipy.sparse import coo_matrix
import HelperMethods as hm
import matplotlib.pyplot as plt

# STEP 0: Define Hyperparameters
gamma = 0.5 # the regularization parameters for X, Y
maxIter = 60 # How many iterations to train the model on
trans = "Normalize" # whether to normalize column data
lss = "Quadratic" # The type of loss to use when training
rank = 5 # the rank of the reconstructed matrix XY
maxRT = 3600*2 # The mAximum runtime of the Model, in seconds
sampSize = 4000 # the size of the sample square sparse matrix
chunkWidth = 5000 # the number of columns to train on



# STEP 1: Initialize and remove any clusters already running, aka clean slate
h2o.init()
h2o.remove_all() 

# STEP 2: Define the model.
# An m x n matrix A, is decombposed into 2 matrices: X[m,k], and Y[k,n]. The goal is to
# minimizer the square difference |A - XY|^2 , where |C| represents the frobenius length of C
# k - the rank of the approximated matrix, Quadratic loss, 
# gamma - the regularization paramaters for the X and Y
# maxIter - how many iterations to perform

# model = H2OGeneralizedLowRankEstimator(k=rank, loss="Quadratic", gamma_x=gamma, gamma_y=gamma, max_iterations=maxIter, transform = trans,
                                       #max_runtime_secs = maxRT)

# STEP 3: Import the sparse matrix, and convert it to an h2o file frame data type
# The sparse matrix contains the training data, where each row is a student, and
# each column is a student answer to a question. an entry at index [i][j]  of '1' means
# the student has answered the question correctly, an entry of '-1' means the student
# answered incorrectly, and 0 means no entry.

# load the csv file located at the same folder level as this file into np array Data
# default - FALSE arg
Data = hm.loadData()
# use buildSampleMatrix() to test the training model with a small sparse matrix
#Data = hm.buildSampleMatrix(sampSize)

# get the size of the data
numDataRows, numDataCols = Data.shape
# calculated the number of column chunks to patition the data into
numChunks = int(np.ceil(numDataCols/chunkWidth))


# STEP 4: partition the data and train the models for each partition
for chunk in range(numChunks):
    # Calculate the starting and ending index at each chunk
    startIdx = chunk*chunkWidth
    endIdx = startIdx + chunkWidth

    # define the chunk from Data
    if (chunk != (numChunks - 1)):
        dataChunk = Data[:, startIdx:endIdx]  ##### DEBUG, change the row setting to all rows when running final script
    else:
        dataChunk = Data[:, startIdx:]

    # convert array to data frame
    dataChunk = pd.DataFrame(dataChunk)

    # convert the dataframe to an h2o Frame
    dataChunk = h2o.H2OFrame(dataChunk)
    dataChunk.describe()
    print("Converted chunk number " + str(chunk) + " out of " + str(numChunks))
    
    model = H2OGeneralizedLowRankEstimator(k=rank, loss="Quadratic", gamma_x=gamma, gamma_y=gamma, max_iterations=maxIter, transform = trans,
                                       max_runtime_secs = maxRT)
    
    model.train(training_frame=dataChunk)
    model.show()
    print("Finished training model for chunk " + str(chunk) + " out of " + str(numChunks))

    # OPTIONAL: plot the objective function score at each iteration
    model_score = model.score_history()
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Objective Function Value per Iteration at chunk: " + str(chunk))
    print (model_score)
    plt.plot(model_score["iterations"], model_score["objective"])
    plt.show()


    # STEP 5: Recover the X and Y features and save them into csv File
    # Idk why its not so straightforward to Recover X and Y but this works...
    # The outputs are X, a numRows x rank Array, and Y a rank x numCols

    Y = model.proj_archetypes(dataChunk)
    x_key = model._model_json["output"]["representation_name"]
    X = h2o.get_frame(x_key)
     
    Y = h2o.as_list(Y)
    X = h2o.as_list(X)
     
    Y.to_csv('outputY_' + str(chunk) + '.csv', index = False)
    X.to_csv('outputX_' + str(chunk) + '.csv', index = False)
  
# Shut down the cluster after use    
h2o.shutdown(prompt=False)


# The rest of this section is commented out code that might help in the future but is no longer needed

# # STEP 5: Impute missing data from X and Y
# pred = model.predict(Data)
# pred.head()
# # converts the prediction h2o frame back into pandas dataframe
# predDataFrame = h2o.as_list(pred)
# # converts prediction dataframe into a numpy array
# predArray = predDataFrame.to_numpy()
# # Shut down the cluster once finished using it
# h2o.shutdown(prompt=False)

# # Outputs the prediction dataframe into a csv file
# predDataFrame.to_csv('outPutSparseMat.csv',index=False)

#to do:
    #change so thjat prediction is dor product of 2 vectos, rather then recunstruxcting all of A

# These comments refer to the sample implementation provided in h2o's GLRM website
# h2o.init()

# # Import the USArrests dataset into H2O:
# arrestsH2O = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/pca_test/USArrests.csv")

# # Split the dataset into a train and valid set:
# train, valid = arrestsH2O.split_frame(ratios=[.8], seed=1234)

# # Build and train the model:
# glrm_model = H2OGeneralizedLowRankEstimator(k=4,
#                                             loss="quadratic",
#                                             gamma_x=0.5,
#                                             gamma_y=0.5,
#                                             max_iterations=700,
#                                             recover_svd=True,
#                                             init="SVD",
#                                             transform="standardize")
# glrm_model.train(training_frame=train)
