#This file generates automatically input files and sizes
#translating EDF files from database (full dataset for patient = 1)
#P03
import pyedflib
import numpy as np
from matplotlib import pyplot as plt
import os
#import pandas as pd
import re 

def get_sizes_train(X, dataset_size, initial_file, dataset):
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
        print(("Reading:database/chb0{}_{}.edf ...".format(dataset,j)))
        f = pyedflib.EdfReader("../database/chb0{}_{}.edf".format(dataset,j))
        n = f.signals_in_file
        input_size = n
        length = f.getNSamples()[0]      
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i,:] = f.readSignal(i)
        
        if length < 921600:
            print('reshaping sizes')
            sigbufs_test = np.pad(sigbufs,((0,0),(0,921600-length)), 'constant', constant_values = (0))
            print('new shape is', np.shape(sigbufs_test))
        elif length > 921600:
            print('reshaping sizes')
            print('SIZE', np.shape(sigbufs_test))
            sigbufs_test = sigbufs_test[0:23,0:921600]
        else:
            sigbufs_test = sigbufs
        X[j-initial_file,0,0,:,:,0]=sigbufs_test
    #max_value = np.amax(abs(X))
    #X = X/max_value
    print(length)
    return X, input_size, length


#Generating target file for the COMPLETE DATA SET (=7)

def target_gen(output, dataset_size, batch_size, seq_num, timesteps, file):
    target = np.zeros((seq_num*(dataset_size), output))
    filepath = file
    dataset = 0
    total_size = batch_size*seq_num*timesteps
    with open(filepath) as fp:
        for cnt, line in enumerate(fp): 
            if line.startswith('File Name:'):
                file_num = re.findall(r'\d+',line)
                dataset = int(file_num[1]) 
            if dataset == dataset_size + 1:   #Finish when data set size is achieved
                break                                   
            if line.startswith('Number of Seizures in File: 1'):
                start_time = re.findall(r'\d+',fp.readline())
                end_time = re.findall(r'\d+',fp.readline())

                batch_start = int(start_time[0])*256
                batch_end =   int(end_time[0])*256

                batch_onset_start = round(((int(start_time[0])*256)/100000)-0.5)*100000 
                print('BATCH ONSET START', batch_onset_start)
                test= ((int(start_time[0])*256) - batch_onset_start)
                print('TEST', test)
                
                onset_start = seq_num*(dataset-1) + round((((int(start_time[0])*256) - batch_onset_start)/(timesteps*batch_size)) - 0.5)
                onset_end = seq_num*(dataset-1) +   round((((int(end_time[0]  )*256) - batch_onset_start)/(timesteps*batch_size)) + 0.5)
                if onset_end > seq_num*dataset:
                    onset_end = onset_end - (onset_end - seq_num*dataset)

                print('DATASET', dataset)
                print('BATCH_START', batch_onset_start)
                print('ONSET_START', onset_start)
                print('ONSET_END', onset_end)
                print('INTER ICTAL START', seq_num*(dataset-1))
                #ictal
                target[onset_start-1:onset_end,1] = 1
                #pre-ictal
                target[seq_num*(dataset-1):onset_start-1,0] = 1
                
                #inter-ictal
                target[onset_end:seq_num*(dataset-1)+int(total_size/(timesteps*batch_size)),2] = 1
                
            if line.startswith('Number of Seizures in File: 0'):


                initial = seq_num*(dataset-1)
                end = seq_num*(dataset-1)+int(total_size/(timesteps*batch_size))

                #non-ictal
                target[initial:end,2] = 1
    return target

    
def get_sizes_test(X, dataset_size, initial_file, dataset):
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
        print(("Reading:database/chb0{}_{}.edf ...".format(dataset,j)))
        f = pyedflib.EdfReader("../database/chb0{}_{}.edf".format(dataset,j))
        n = f.signals_in_file
        input_size = n
        length = f.getNSamples()[0]      
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i,:] = f.readSignal(i)
        
        if length < 921600:
            print('reshaping sizes')
            sigbufs_test = np.pad(sigbufs,((0,0),(0,921600-length)), 'constant', constant_values = (0))
            print('new shape is', np.shape(sigbufs_test))
        elif length > 921600:
            print('reshaping sizes')
            print('SIZE', np.shape(sigbufs_test))
            sigbufs_test = sigbufs_test[0:23,0:921600]
        else:
            sigbufs_test = sigbufs
        X[j-initial_file,0,0,:,:,0]=sigbufs_test
    #max_value = np.amax(abs(X))
    #X = X/max_value
    print(length)
    return X, input_size, length
    
    
    
def batching_dataset_02(dataset_size, extra_data_set, timesteps, seq_number, batch_size_new, input_size_new, X ):
    X_new=np.zeros((seq_number*dataset_size, timesteps, input_size_new, batch_size_new, 1))
    for m in range(0,dataset_size-extra_data_set):
        if (m == 1-1 or  m == 2-1 or m == 3-1 or m == 4-1):
            if m == 1-1:
                initial_time = 0
            if m == 2-1:
                initial_time = 100000
            if m == 3-1:
                initial_time = 100000
            if m == 4-1:
                initial_time = 500000
        else:
               initial_time = 0
        print('initial time:', initial_time)
        for i in range(0,seq_number):
            for j in range(0,timesteps):
                initial = initial_time+(i*batch_size_new*timesteps)+j*batch_size_new
                final = initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new)
                X_new[seq_number*m+i,j,:,0:batch_size_new,0] =  X[m,0,0,0:input_size_new,initial:final,0] 
    return X_new

    
def batching_dataset_01(dataset_size, extra_data_set, timesteps, seq_number, batch_size_new, input_size_new, X ):
    X_new=np.zeros((seq_number*dataset_size, timesteps, input_size_new, batch_size_new, 1))
    for m in range(0,dataset_size-extra_data_set):
        if (m == 3-1 or  m == 4-1 or m == 15-1 or m == 16-1 or m == 18-1 ):
            if m == 3-1:
                initial_time = 700000
            if m == 4-1:
                initial_time = 300000
            if m == 15-1:
                initial_time = 400000
            if m == 16-1:
                initial_time = 200000
            if m == 18-1:
                initial_time = 400000
        else:
               initial_time = 0
        print('initial time:', initial_time)
        for i in range(0,seq_number):
            for j in range(0,timesteps):
                initial = initial_time+(i*batch_size_new*timesteps)+j*batch_size_new
                final = initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new)
                X_new[seq_number*m+i,j,:,0:batch_size_new,0] =  X[m,0,0,0:input_size_new,initial:final,0]     
    return X_new
    
    












