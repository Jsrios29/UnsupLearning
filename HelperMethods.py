# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 19:49:05 2020

This file contains all the helper methods needed to perform in GLRM.py

@author: Juan Rios
"""


import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import random
import scipy.sparse
from scipy import sparse

# loadData() loads the data from the local folder, and returns a sparse matrix representation
# chunk - a boolean if true: load a small chunk of the data for test purposes, false to load entire data
def loadData():
    
    # Load the original Training Data
    Data = pd.read_csv(r"train_task_1_2_small.csv")
    
    # Convert the dataframe Data, into a numpy array
    # each row contains a QuestionID, UserId, AnswerId, isCorrect, CorrectAnswer, and AnswerValue
    # is Correct is 1 if true and 0 if false, however to build the sparse matrix, convert 0 to -1, 
    # and let 0 denote unkown values
    Data = Data.to_numpy()
    


    # replace isCorrect entries that are 0 to -1, so sparse array can have '0' as unknown value
    isCorrect = Data[:,3]
    isCorrect[isCorrect == 0] = -1

    # load coo_matrix from Scipy.sparse module
    # COO Matrices allow us to store each row as a list of 3 elements: row idx, col idx, and data.
    # for our purposes, this means UserId, QuestionId, and isCorrect
    from scipy.sparse import coo_matrix


    UserId  = np.array(Data[:,1])
    QuestionId  = np.array(Data[:,0])

    # Build the COO Matrix, convert the matrix to 2-D array, and return
    B = coo_matrix((isCorrect, (UserId, QuestionId)))
    #B = B.toarray()
    
   
    # B = pd.DataFrame(B)  # 1st row as the column names
    return B





# This method builds a sparse matrix of size numRows x numCols
# an the known values are 1 and -1, a 0 means the value is unknown
# sometimes values are '2' nut its ok for testing purposes
def buildSampleMatrix(numRows):

    # generate a vector of length 10*numRows, where each element in the vector
    # represents a row index, uniformly generated from 0 to the numRows
    possRowIdx = np.random.uniform(0, numRows, size = 10*numRows)
    possRowIdx = (np.floor(possRowIdx)).astype(np.int64)
    # Same as possRowIdx buit with column indeces
    possColIdx = np.random.uniform(0, numRows, size = 10*numRows)
    possColIdx = (np.floor(possColIdx)).astype(np.int64)
    
    # generate the data to be -1, 0, or 1, where approx ~10 will be -1 or 1
    data = np.random.uniform(-1,1,size = 10*numRows)
    data = (np.round(data)).astype(np.int64)
    
    # build the sparse matrix
    sparseMat = sparse.coo_matrix((data, (possRowIdx, possColIdx)))
    
    sparseDataFrame = pd.DataFrame(sparseMat.toarray())
    
    return sparseDataFrame