# UnsupLearning
Various unsupervised learning techniques for predicting student responses to online learning environment Eedi

Description: This project used various unsupervised techniques for predicting student correctness to questions featured on the online learning platform Eedi. I work with a fellow ML friend Shri Shriddhar, I was in charge of implementing the Generalized Low Rank Model (GLRM) and the restricted Boltzmann Machine. Shri worked with Singular Value Decomposition techniques.

Files:
1. GLRM_classifier.py - takes a trained model and classifies the student data
2. HelperMethods.py - contains methods that aid in loading data
3. RBM.py - contains the main development, training, and testing of the Restricted Boltzmann Machine
4. RBMVisualizer.mlx - matlab livescript file that visualises the results of the RBM
5. RBM_results.txt - contains results for various runs of the RBM, with a description of parameters and confusion matrix
6. Results Summary.txt - results for the GLRM
7. Unsup_learning_final_Rios_Shridhar.pdf - contains all the information regarding the approach and results of the project
8. dataPreprocess.py -  deals with handling the raw dating and storing it into a sparse matrix
9. miniGLRM_classfier.py - a classifier that handles smaller data, as in the main data is broken into chunks.
10. MiniGLRM_collab.py - an attempt to develop the GLRM on google collab
