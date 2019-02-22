# imports

import numpy as np
import re
from random import randint
import target_data_gen
from target_data_gen import get_sizes
from target_data_gen import target_gen

import model_hard_sigmoid
from model_hard_sigmoid import GRU, Conv2D, MaxPool2D, sigmoid, reLU, tanh, CrossEntropy, Optimizer, orthogonal_initializer

# ***************************************#

print("Preparing for test phase ...")
startfile = 9
nb_files = 9
dataset_size = 18
seq_number = 100
batch_size = 100
timesteps = 10
conv_filters = 2
conv_height = 2
conv_width = 2
inputs = 23
GRUoutputs = 100
GRUinputs = int((inputs-1)/2)*int((batch_size-1)/2)*conv_filters

#Defining sizes for input/target data
X_test=np.zeros((seq_number, timesteps, inputs, batch_size, 1))
layer_0 = Conv2D(conv_height,conv_width,conv_filters)
# add activation
layer_1 = MaxPool2D()
layer_2 = GRU(seq_number, timesteps, GRUinputs, GRUoutputs)

#init empty arrays (hardcoded for now)
#Wconv_flat = np.zeros((4,2))
Wconv_flat, Bconv, Wz, Wr, Wh, Uz, Ur, Uh, Bz, Br, Bh, Wlin, Blin = [],[],[],[],[],[],[],[],[],[],[],[],[]
#Bconv = np.zeros((1,2))

#Wz, Wr, Wh = np.zeros((GRUinputs,GRUoutputs)), np.zeros((GRUinputs,GRUoutputs)), np.zeros((GRUinputs,GRUoutputs))
#Uz, Ur, Uh = np.zeros((GRUoutputs,GRUoutputs)), np.zeros((GRUoutputs,GRUoutputs)), np.zeros((GRUoutputs,GRUoutputs))
#Bz, Br, Bh = np.zeros((1, GRUoutputs)), np.zeros((1, GRUoutputs)), np.zeros((1, GRUoutputs))
#Wlin = np.zeros((GRUoutputs,1))
#Blin = np.zeros((1,1))

def parse_state(line, string):
    if line == string:
        return 1
    else:
        return 0

def fill_array(line, array, state):
    line = re.sub('[\n]', '', line)
    if state == True:
        array.append(line.split(","))
        return 1
    else:
        return 0

        
#load from textfiles
BoolWconv, BoolBconv,BoolWz, BoolWr, BoolWh, BoolUz, BoolUr, BoolUh, BoolBz, BoolBr, BoolBh, BoolWlin, BoolBlin = False, False, False, False, False, False, False, False, False, False, False, False, False
with open("all_param_dummy.txt") as file_Wall:
    for line in file_Wall:
        skip = parse_state(line, "Wconv\n")
        if skip == 1:
            BoolWconv = True
            continue
        skip = parse_state(line, "Bconv\n")
        if skip == 1:
            BoolWconv, BoolBconv = False, True           
            continue
        skip = parse_state(line, "Wz\n")
        if skip == 1:
            BoolBconv, BoolWz = False, True
            continue
        skip = parse_state(line, "Wr\n")
        if skip == 1:
            BoolWz, BoolWr = False, True
            continue
        skip = parse_state(line, "Wh\n")
        if skip == 1:
            BoolWr, BoolWh = False, True
            continue
        skip = parse_state(line, "Uz\n")
        if skip == 1:
            BoolWh, BoolUz = False, True
            continue
        skip = parse_state(line, "Ur\n")
        if skip == 1:
            BoolUz, BoolUr = False, True
            continue
        skip = parse_state(line, "Uh\n")
        if skip == 1:
            BoolUr, BoolUh = False, True                           
            continue
        skip = parse_state(line, "Bz\n")
        if skip == 1:
            BoolUh, BoolBz = False, True
            continue
        skip = parse_state(line, "Br\n")
        if skip == 1:
            BoolBz, BoolBr = False, True
            continue
        skip = parse_state(line, "Bh\n")
        if skip == 1:
            BoolBr, BoolBh = False, True
            continue
        skip = parse_state(line, "Wlin\n")
        if skip == 1:
            BoolBh, BoolWlin = False, True
            continue
        skip = parse_state(line, "Blin\n")
        if skip == 1:
            BoolWlin, BoolBlin = False, True
            continue
        skip = fill_array(line, Wconv_flat, BoolWconv)
        if skip == 1:
            continue
        skip = fill_array(line, Bconv, BoolBconv)
        if skip == 1:
            continue
        skip = fill_array(line, Wz, BoolWz)
        if skip == 1:
            continue
        skip = fill_array(line, Wr, BoolWr)
        if skip == 1:
            continue
        skip = fill_array(line, Wh, BoolWh)
        if skip == 1:
            continue
        skip = fill_array(line, Uz, BoolUz)
        if skip == 1:
            continue
        skip = fill_array(line, Ur, BoolUr)
        if skip == 1:
            continue
        skip = fill_array(line, Uh, BoolUh)
        if skip == 1:
            continue
        skip = fill_array(line, Bz, BoolBz)
        if skip == 1:
            continue
        skip = fill_array(line, Br, BoolBr)
        if skip == 1:
            continue
        skip = fill_array(line, Bh, BoolBh)
        if skip == 1:
            continue
        skip = fill_array(line, Wlin, BoolWlin)
        if skip == 1:
            continue
        skip = fill_array(line, Blin, BoolBlin)
        if skip == 1:
            continue

Wconv_flat = np.asarray(Wconv_flat, dtype = np.float64)
Bconv = np.asarray(Bconv, dtype = np.float64)
Wz = np.asarray(Wz, dtype = np.float64)
Wr = np.asarray(Wr, dtype = np.float64)
Wh = np.asarray(Wh, dtype = np.float64)
Uz = np.asarray(Uz, dtype = np.float64)
Ur = np.asarray(Ur, dtype = np.float64)
Uh = np.asarray(Uh, dtype = np.float64)
Bz = np.asarray(Bz, dtype = np.float64)
Br = np.asarray(Br, dtype = np.float64)
Bh = np.asarray(Bh, dtype = np.float64)
Wlin = np.asarray(Wlin, dtype = np.float64)
Blin = np.asarray(Blin, dtype = np.float64)
Wconv = Wconv_flat.reshape(2,2,2)
layer_0.set_param(Wconv, Bconv)
layer_2.set_param(Wz, Wr, Wh, Uz, Ur, Uh, Bz, Br, Bh, Wlin, Blin)
X, Y = list(), list()
X, input_size, length = get_sizes(X, dataset_size) #Defining sizes for input/target data


for file_iter in range(startfile, nb_files+startfile):
    print("Reading files ", startfile+1, " to ", nb_files + startfile, "\n")
    m=file_iter
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
            initial = initial_time+(i*batch_size*timesteps)+j*batch_size
            final = initial_time+(i*batch_size*timesteps)+((j+1)*batch_size)
            #print(initial_time+(i*batch_size_new*timesteps)+j*batch_size_new)
            #print(initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new))
            X_test[i,j,:,0:batch_size,0] =  X[m,0,0,0:inputs,initial:final,0]    
    print("Normalizing data ... ")  
    max_value_2 = np.amax(abs(X_test))
    X_test = X_test/max_value_2
    print("Generating predictions ...")
    HConv = layer_0.forward(X_test) # 5D data for train_data, 3D for Wconv 2D for Bconv
    print("HCONV", HConv[0:5, 0,0, 0, 0])
    YConv = reLU(HConv, deriv=False) # no requirement on shape
    print("YCONV", YConv[0:5, 0,0, 0, 0])
    #pooling layer
    pool_kernel = np.array([2,2])
    YPool, XArgmax = layer_1.forward(YConv,  pool_kernel) #5D data for YConv
    #flattening
    X_GRU_flat = YPool.reshape(YPool.shape[0],10,-1) # check size here should be 3D (100*nb_files, 10, 1078)
    #print(X_GRU_flat.shape)
    #GRU
    yhat_test = layer_2.forward(X_GRU_flat) # timesteps
    file2 = open("../results_post_separation/{}_numPy_HS_WQon_ep40.txt".format(file_iter+1), 'w')
    np.savetxt(file2, yhat_test, delimiter="," )
    file2.close()

    print("Predictions saved to file : ", file2)