# imports

import numpy as np
import re
from random import randint
import target_data_gen_100seq_GENERAL
from target_data_gen_100seq_GENERAL import get_sizes_test
from target_data_gen_100seq_GENERAL import target_gen
import matplotlib.pyplot as plt

# ***************************************#

startfile = 18
nb_files = 8
dataset_size = 26
nbseq = 100
batch_size = 100
timesteps = 10
conv_filters = 2
conv_height = 2
conv_width = 2
inputs = 23
GRUoutputs = 50
GRUinputs = int((inputs-1)/2)*int((batch_size-1)/2)*conv_filters


# 2D matrices
def error_count(y, yhat, startfile, nb_files, nbseq):
    MAE, MSE, MRAE = [],[],[]
    for i in range(startfile, nb_files+startfile):
        Y = y[i*nbseq:(i+1)*nbseq]
        YHAT = yhat[(i-startfile)*nbseq:(i+1-startfile)*nbseq]
        total = nbseq*Y.shape[1]
        MAE.append(np.sum(np.abs(Y - YHAT))/total)
        MSE.append(np.sum(np.power(Y-YHAT, 2))/(2*total))
        p = np.abs(np.argmax(YHAT, axis = 1) - np.argmax(Y, axis = 1))
        p[p == 2] = p[p==2]/2
        MRAE.append(np.sum(p)/nbseq)
    return MAE, MSE, MRAE

def plot_res(y, yhat,nbseq):
    for i in range(startfile, nb_files+startfile):
        
        Y = y[i*nbseq:(i+1)*nbseq]
        YHAT = yhat[(i-startfile)*nbseq:(i+1-startfile)*nbseq]
        x = np.arange(nbseq)
        fig1= plt.figure()
        plt.plot(x, np.argmax(YHAT, axis = 1))
        fig1.suptitle(" Predictions file {}".format(i+1))
        plt.show()
        fig2 = plt.figure()
        plt.plot(x, np.argmax(Y, axis = 1))
        fig2.suptitle(" Ground truth file {}".format(i+1))
        plt.show()
    return
        
########################################


targets, predictions = [], []
if batch_size == 50 and nbseq == 100:
    gt_file = open("../database/chb01-targets_50batch.txt", 'r')
    print("reading: ../database/chb01-targets_50batch.txt \n")
    
if batch_size == 100 and nbseq == 100:
    gt_file = open("../database/chb01-targets", 'r')
    print("reading: ../database/chb01-targets \n")
    
if batch_size == 200 and nbseq == 100:
    gt_file = open("../database/chb01-targets_200batch.txt", 'r')
    print("reading: ../database/chb01-targets.txt \n")

if batch_size == 400 and nbseq == 50:
    gt_file = open("../database/chb01-targets_50seq.txt", 'r')
    print("reading: ../database/chb01-targets_50seq.txt \n")
for line in gt_file:
    line = re.sub('\n','', line)
    targets.append(line.split(","))


#for file_iter in range(startfile, nb_files+startfile):
    #print("Reading: ../results_3class_kerasTrain/{}_numPy_HS_WQon_ep40.txt \n".format(file_iter+1))
    #with open("../results_3class_kerasTrain/{}_numPy_HS_WQon_ep40.txt".format(file_iter+1), 'r') as resfile:
print("Reading: ../results_3class_fullKeras/test_19to26_45ep_noUPSAMPLING.txt")
with open("../results_3class_fullKeras/test_19to26_45ep_noUPSAMPLING.txt") as resfile:
    for line in resfile:
        if line.startswith(' Confusion Matrix'):
            break
        else:
            line = re.sub('\n','', line)
            predictions.append(line.split(","))

targets = np.asarray(targets, dtype = np.float16)
predictions = np.asarray(predictions, dtype = np.float16)

MAE, MSE, MRAE = error_count(targets, predictions, startfile, nb_files, nbseq)
for file_iter in range(startfile, nb_files+startfile):
    print("MAE for file {} -------  {}".format(file_iter+1, MAE[file_iter-startfile]))
    print("MSE for file {} -------  {}".format(file_iter+1, MSE[file_iter-startfile]))
    print("MRAE for file {} -------  {}".format(file_iter+1, MRAE[file_iter-startfile]))

plot_res(targets, predictions, nbseq)

    