# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 22:58:38 2020

The purpose of this script is to load the complete data, divide it into 6 sections, and form 6 sets
each set has 100,000 training examples and about 20,000 validation examples.

the data is stored in the Data folder in the directory of this file

@author: Juan Rios
"""

# %% 1. Importing Libraries and Data
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as ss
from scipy.sparse import coo_matrix

# load the complete data set
Data = Data = pd.read_csv(r"/Users/juanrios/Spyder/539 Final Project/Data/train_task_1_2.csv")
# convert data to numpy
Data = Data.to_numpy()

# replace isCorrect entries that are 0 to 1, , and 1 to 2, and thus 0 menas no entries in the sparse arr
isCorrect = Data[:,3]
# isCorrect[isCorrect == 1] = 2
isCorrect[isCorrect == 0] = -1

# define the rows, UserId
UserId  = np.array(Data[:,1])
# define the columns, QuestionId
QuestionId  = np.array(Data[:,0])

# split into t1, t2 ... t6 training sets and v1, v2, ... v6 validation sets

Data = np.zeros((UserId.shape[0],3))
Data[:,0] = QuestionId
Data[:,1] = UserId
Data[:,2] = isCorrect

s1 = Data[0:3173570,:].astype(np.int64)
s2 = Data[3173570:3173570*2,:].astype(np.int64)
s3 = Data[3173570*2:3173570*3,:].astype(np.int64)
s4 = Data[3173570*3:3173570*4,:].astype(np.int64)
s5 = Data[3173570*4:,:].astype(np.int64)

# %% 2. create the 5 deifferent training/validation sets and save them to data folder

t1 = np.concatenate((s1,s2,s3,s4), axis = 0)
t1 = coo_matrix((t1[:,2], (t1[:,1], t1[:,0])))
v1 = pd.DataFrame(s5)
ss.save_npz(r'Data/t1.npz', t1)
v1.to_csv(r'Data/v1.csv', index = False)

t2 = np.concatenate((s1,s2,s3,s5), axis = 0)
t2 = coo_matrix((t2[:,2], (t2[:,1], t2[:,0])))
v2 = pd.DataFrame(s4)
ss.save_npz(r'Data/t2.npz', t2)
v2.to_csv(r'Data/v2.csv', index = False)

t3 = np.concatenate((s1,s2,s4,s5), axis = 0)
t3 = coo_matrix((t3[:,2], (t3[:,1], t3[:,0])))
v3 = pd.DataFrame(s3)
ss.save_npz(r'Data/t3.npz', t3)
v3.to_csv(r'Data/v3.csv', index = False)

t4 = np.concatenate((s1,s3,s4,s5), axis = 0)
t4 = coo_matrix((t4[:,2], (t4[:,1], t4[:,0])))
v4 = pd.DataFrame(s2)
ss.save_npz(r'Data/t4.npz', t4)
v4.to_csv(r'Data/v4.csv', index = False)


t5 = np.concatenate((s2,s3,s4,s5), axis = 0)
t5 = coo_matrix((t5[:,2], (t5[:,1], t5[:,0])))
v5 = pd.DataFrame(s1)
ss.save_npz(r'Data/t5.npz', t5)
v5.to_csv(r'Data/v5.csv', index = False)






