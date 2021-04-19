#!/usr/bin/env python
# coding: utf-8

# # mini Generalized Low Rank Model

# The purpose of this script is to run GLRM on a chunk of the total sparse matrix. The total sparse matrix 120k x 27k, while this code is meant to test running the model on a sub matrix, 120K x 5k, which results in lower computational time. additionally, our Assumption is that 5k Questions is enough to give underlying patterns that could achieve reasonable prediction, rather than attmepting to find a low rank matrix for the full sized data.

# In[1]:


import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2OGeneralizedLowRankEstimator
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import random
import scipy.sparse
from scipy import sparse
from numba import jit, cuda 


# In[2]:


# In[3]:


# STEP 0: Define Hyperparameters
gamma = 0.5 # the regularization parameters for X, Y
maxIter = 150 # How many iterations to train the model on
trans = "Normalize" # whether to normalize column data
lss = "Quadratic" # The type of loss to use when training
rank = 100 # the rank of the reconstructed matrix XY
maxRT = 3600*13 # The maximum runtime of the Model, in seconds
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

model = H2OGeneralizedLowRankEstimator(k=rank, loss="Quadratic", gamma_x=gamma, gamma_y=gamma, max_iterations=maxIter, transform = trans,
                                       max_runtime_secs = maxRT)


# In[4]:


# STEP 3: Import the sparse matrix, and convert it to an h2o file frame data type
# The sparse matrix contains the training data, where each row is a student, and
# each column is a student answer to a question. an entry at index [i][j]  of '1' means
# the student has answered the question correctly, an entry of '-1' means the student
# answered incorrectly, and 0 means no entry.

# load the csv file located at the same folder level as this file into np array Data (first 5000 Cols)
Data = sparse.load_npz("first_5000_answers.npz")
Data = Data.toarray()


# In[5]:


# convert array to data frame
Data = pd.DataFrame(Data)

# convert the dataframe to an h2o Frame
Data = h2o.H2OFrame(Data)


# In[6]:

 
model.train(training_frame=Data)
model.show()
print("Finished training model")


# In[10]:


# OPTIONAL: plot the objective function score at each iteration
model_score = model.score_history()
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.title("Objective Function Value per Iteration")
print (model_score)
plt.plot(model_score["iterations"], model_score["objective"])
plt.show()
plt.savefig('modelScore.jpg')

# STEP 5: Recover the X and Y features and save them into csv File
# Idk why its not so straightforward to Recover X and Y but this works...
# The outputs are X, a numRows x rank Array, and Y a rank x numCols

Y = model.proj_archetypes(Data)
x_key = model._model_json["output"]["representation_name"]
X = h2o.get_frame(x_key)
     
Y = h2o.as_list(Y)
X = h2o.as_list(X)
     
Y.to_csv('outputY.csv', index = False)
X.to_csv('outputX.csv', index = False)
  
# Shut down the cluster after use    
h2o.shutdown(prompt=False)


# In[ ]:
