#This file generates automatically input files and sizes
#translating EDF files from database (full dataset for patient = 1)
import pyedflib
import numpy as np
from matplotlib import pyplot as plt
import os
#import pandas as pd
import re 

#defining parameters
output = 1
#X=np.array(sigbufs)
batch_size = 10000 #(length)
batch_mod = 100
dataset_size = 7



def get_sizes(X, dataset_size, initial_file):
    # f = pyedflib.EdfReader("../../dataset_01/chb01_{}.edf".format(initial_file))
    # n = f.signals_in_file
    # input_size = n
    # length = f.getNSamples()[0]
    # sigbufs = np.zeros((n, f.getNSamples()[0]))

    # for i in np.arange(n):
            # sigbufs[i,:] = f.readSignal(i)

    # sigbufs_test = sigbufs
    # print('SIZE sigbufs', np.shape(sigbufs_test))
    length = 921600
    input_size = 23
    X=np.zeros((dataset_size,1,1, input_size, length,1))
    #X[0,0,0,:,:,0]=sigbufs_test 
    for j in range(initial_file, initial_file+dataset_size):
        if (j == 19 or j==28 or j==35 or j==44 or j==45):
            continue
        print(("Reading:dataset_01/chb01_{}.edf ...".format(j)))
        print('J is', j)
        f = pyedflib.EdfReader("../../dataset_01/chb01_{}.edf".format(j))
        n = f.signals_in_file
        input_size = n
        length = f.getNSamples()[0]      
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i,:] = f.readSignal(i)
        sigbufs_test = sigbufs
        if length < 921600:
         print('reshaping sizes')
         sigbufs_test = np.pad(sigbufs_test,((0,0),(0,921600-length)), 'constant', constant_values = (0))
         print('new shape is', np.shape(sigbufs_test))
        X[j-initial_file,0,0,:,:,0]=sigbufs_test
    #max_value = np.amax(abs(X))
    #X = X/max_value
    print(length)
    return X, input_size, length














