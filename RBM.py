# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:01:20 2020

The purpose is to adapt the code found here: https://github.com/srp98/Movie-Recommender-using-RBM/blob/master/Recommender_System.py

The web-link shows an implementation of RBM for movie ratings, we can treat our data

@author: Juan Rios
"""

# %% Import dependencies and hyper-parameters
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse

batchSize = 1000 # number of batches before weights are updated
alpha = 0.01/batchSize # learning rate
hiddenUnits = 50; # recommended 100
epochs = 5; # recommended 40-50


# %% 2. Load the training and validation data

# load tbhe training data for the first 5000 features
TrainData = sparse.load_npz(r"/Users/juanrios/Spyder/539 Final Project/Data/t4.npz") # <----DONT FORGET TO CHange prined out as to not over write1
TrainData = TrainData.tocsr()
# normalize array, 1 = correct, 0.5 = incorrect, 0 = missing
# TrainData = TrainData/2 
# load the validation data
valData = pd.read_csv(r"/Users/juanrios/Spyder/539 Final Project/Data/v4.csv").to_numpy()

#convert the train data to numpy array, and normalize
TrainData = TrainData.toarray()

# %% 3.  setting up the model parameters

# Setting the models Parameters

visibleUnits = TrainData.shape[1]
vb = tf.placeholder(tf.float32, [visibleUnits])  # Number of unique movies
hb = tf.placeholder(tf.float32, [hiddenUnits])  # Number of features were going to learn
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weight Matrix

# Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling

# Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)


# %% 4. setting RBM training parameters


# Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

# Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set the error function, here we use Mean Absolute Error Function
err = v0 - v1
err_sum = tf.reduce_mean(err*err)

# %% 5. training the RBM

# Current weight
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)

# Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)

# Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)

# Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train RBM with 15 Epochs, with Each Epoch using 10 batches with size 100, After training print out the error by epoch


errors = []
for i in range(epochs):
    print("Current epoch: " + str(i))
    for start, end in zip(range(0, TrainData.shape[0], batchSize), range(batchSize, TrainData.shape[0], batchSize)):
        batch = TrainData[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: TrainData, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print(errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show() 

# %% 6. Classify 

# recall that userId is at column index 1
print("Training has finished, and classification is starting")
TP = 0 
FP = 0
TN = 0
FN = 0
counter = 0
valHeight = valData.shape[0]

# recunstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: TrainData, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})


median = np.median(rec)
predictions = np.zeros((valHeight))

for user in valData:
    
    Qidx = user[0]
    Uidx = user[1]
    label = user[2]
    

    pred = rec[Uidx, Qidx]
    predictions[counter] = pred
    
    if pred <= median:
        pred = 0
    else:
        pred = 1
    
    if pred == 0 and label == -1:
            TN = TN + 1
    elif pred == 1 and label == 1:
            TP = TP + 1
    elif pred == 1 and label == -1:
            FP = FP + 1
    elif pred == 0 and label == 1:
            FN = FN + 1
    else:    
        print("Something went wrong with computing prediction rates, prediction and labels are : " + str(pred) + " " + str(label))

    # print out every 1000 rows:
    if (counter % 200000) == 0:
            print("Currently on " + str(counter) + " out of: " + str(valHeight))
        
    counter = counter + 1

print("Final validation rate = " + str((TP + TN)/((TN + TP + FN + FP))))
print("TN = " + str(TN))
print("TP = " + str(TP))
print("FN = " + str(FN))
print("FP = " + str(FP))

predictions = pd.DataFrame(predictions)
predictions.to_csv(r'/Users/juanrios/Spyder/539 Final Project/Data/preds_R4.csv', index = False)