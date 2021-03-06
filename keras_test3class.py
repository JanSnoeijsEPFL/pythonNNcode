# imports

import numpy as np
import re
from random import randint
import target_data_gen_100seq_GENERAL
from target_data_gen_100seq_GENERAL import get_sizes_test
from target_data_gen_100seq_GENERAL import target_gen
import keras
import model_softmax

# ***************************************#

print("Preparing for test phase ...")
startfile = 18
nb_files = 8
dataset_size = 26
seq_number = 100
batch_size = 100
timesteps = 10
conv_filters = 2
conv_height = 2
conv_width = 2
inputs = 23
GRUoutputs = 100
electrodes = 23

GRUinputs = int((inputs-1)/2)*int((batch_size-1)/2)*conv_filters
max_value = 1061.00122101221 # hardcoded max value for data normalization --> to be written in text file from training phase
#Defining sizes for input/target data
X_test=np.zeros((seq_number*nb_files, timesteps, inputs, batch_size, 1))
layer_0 = model_softmax.Conv2D(conv_height,conv_width,conv_filters)
# add activation
#layer_1 = model_softmax.MaxPool2D()
layer_1 = model_softmax.fast_MaxPool2D()
layer_2 = model_softmax.GRU(seq_number*nb_files, timesteps, GRUinputs, GRUoutputs)

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
with open("keras_param_3class_30e.txt") as file_Wall:
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
layer_2.set_param(Wz, Wr, Wh, Uz, Ur, Uh, Bz.reshape(1,100), Br.reshape(1,100), Bh.reshape(1,100), Wlin, Blin.reshape(1,3))

X, Y = list(), list()
patient = 1
X, input_size, length = get_sizes_test(X, dataset_size, 1, patient) #Defining sizes for input/target data


for file_iter in range(startfile, 1+startfile):
    print("Reading files ", startfile+1, " to ", nb_files + startfile, "\n")
    m=file_iter
    if (m == 3-1 or  m == 4-1 or m == 15-1 or m == 16-1 or m == 18-1 or m == 21-1 or m == 26-1):
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
        if m == 21-1:
            initial_time = 0
        if m == 26-1:
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
            X_test[(m-startfile)*seq_number+i,j,:,0:batch_size,0] =  X[m,0,0,0:inputs,initial:final,0]    
    print("Normalizing data ... ")  
X_test = X_test/max_value
print("Generating predictions ...")

model1 = keras.models.Sequential()
model1.add(keras.layers.TimeDistributed(keras.layers.Conv2D(2,(2,2), activation = 'relu'), input_shape = (None,electrodes,batch_size,1)))
model1.layers[0].set_weights((Wconv.reshape(2,2,1,2),Bconv.reshape(2,)))
model1.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size =(2,2))))
model1.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
model1.add(keras.layers.GRU(100))
model1.layers[3].set_weights((np.concatenate((Wz, Wr, Wh), axis = 1), np.concatenate((Uz, Ur, Uh), axis = 1), np.concatenate((Bz.reshape(100,), Br.reshape(100,), Bh.reshape(100,)), axis = 0)))
model1.add(keras.layers.Dense(3, activation = 'softmax'))
model1.layers[4].set_weights((Wlin, Blin.reshape(3,)))
yhat_keras = model1.predict(X_test)


HConv = layer_0.forward(X_test) # 5D data for train_data, 3D for Wconv 2D for Bconv
YConv = model_softmax.reLU(HConv, deriv=False)

#custom_Conv = model2.predict(Keras_YConv)

# no requirement on shape
#pooling layer
pool_kernel = np.array([2,2])
YPool, XArgmax = layer_1.forward(YConv,  pool_kernel) #5D data for YConv
#flattening

X_GRU_flat = YPool.reshape(YPool.shape[0],10,-1) # check size here should be 3D (100*nb_files, 10, 1078)
#print(X_GRU_flat.shape)
#GRU
yhat_test = layer_2.forward(X_GRU_flat) # timesteps
print("total diff post GRU", np.sum(abs(yhat_test-yhat_keras)))
print("max diff post GRU", np.max(abs(yhat_test - yhat_keras)))
#for i in range(timesteps):
 #   print("diff GRU in iteration {}: ".format(i), np.sum(abs(yhat_test[:,i,:]- yhat_keras[:,i,:])))
print("shapes", yhat_test.shape, yhat_keras.shape)
file2 = open("../results_3class_fullKeras/test_customFP_kerasTrain_30e.txt", 'w')
np.savetxt(file2, yhat_test, delimiter="," )
file2.close()

print("Predictions saved to file : ", file2)
fileTarget = open("../database/chb01-targets", 'r')
YtestTrue  = np.loadtxt(fileTarget, delimiter = ",")
fileTarget.close()
YtestTrue = YtestTrue[startfile*seq_number:(startfile+nb_files)*seq_number,:]

goodPreIctal = 0
goodIctal = 0
goodHealthy = 0
countPreIctal = 0
countIctal = 0
countHealthy = 0
IctalAsPreIctal = 0
IctalAsHealthy = 0
PreIctalAsHealthy = 0
PreIctalAsIctal = 0
HealthyAsPreIctal = 0
HealthyAsIctal = 0
Y_stat = np.argmax(yhat_test, axis = 1)
yhat_test = keras.utils.to_categorical(Y_stat, 3)


for k in range(YtestTrue.shape[0]):
    if YtestTrue[k, 0] == 1:
        countPreIctal += 1
        if yhat_test[k, 0] == 1:
            goodPreIctal += 1 
        elif yhat_test[k,1] == 1:
            PreIctalAsIctal += 1
        else:
            PreIctalAsHealthy += 1

    elif YtestTrue[k, 1] ==  1:
        countIctal += 1
        if yhat_test[k,1] == 1:
            goodIctal += 1
        elif yhat_test[k,0] == 1:
            IctalAsPreIctal += 1
        else:
            IctalAsHealthy +=1

    elif YtestTrue[k, 2] ==  1:
        countHealthy +=1
        if yhat_test[k,2] == 1:
            goodHealthy +=1
        elif yhat_test[k,0] == 1:
            HealthyAsPreIctal +=1
        else:
            HealthyAsIctal +=1

TestSF = open("../results_3class_fullKeras/test_customFP_kerasTrain_30e.txt", 'a')
TestSF.write(" Confusion Matrix\n")
TestSF.write("             | Inter-Ictal | Pre-Ictal | Ictal\n")
TestSF.write("----------------------------------------------\n")
TestSF.write(" Inter-Ictal |     {}      |    {}     |  {}  \n".format(goodHealthy, PreIctalAsHealthy, IctalAsHealthy))
TestSF.write("----------------------------------------------\n") 
TestSF.write(" Pre-Ictal   |     {}      |    {}     |  {}  \n".format(HealthyAsPreIctal, goodPreIctal, IctalAsPreIctal))
TestSF.write("----------------------------------------------\n") 
TestSF.write(" Ictal       |     {}      |    {}     |  {}  \n".format(HealthyAsIctal, PreIctalAsHealthy, goodIctal))
TestSF.write("\n")
TestSF.write("Correctly classified inter-ictals: {}/{} = {}%\n".format(goodHealthy, countHealthy, goodHealthy/countHealthy*100))
TestSF.write("Correctly classified pre-ictals  : {}/{} = {}%\n".format(goodPreIctal, countPreIctal, goodPreIctal/countPreIctal*100))
TestSF.write("Correctly classified ictals      : {}/{} = {}%\n".format(goodIctal, countIctal, goodIctal/countIctal*100))
TestSF.write("\n")
TestSF.write(" \n")

goodPreIctal = 0
goodIctal = 0
goodHealthy = 0
countPreIctal = 0
countIctal = 0
countHealthy = 0
IctalAsPreIctal = 0
IctalAsHealthy = 0
PreIctalAsHealthy = 0
PreIctalAsIctal = 0
HealthyAsPreIctal = 0
HealthyAsIctal = 0
Y_stat = np.argmax(yhat_keras, axis = 1)
yhat_keras = keras.utils.to_categorical(Y_stat, 3)
print("custom - keras err: ", np.sum(abs(yhat_test - yhat_keras)))
for k in range(YtestTrue.shape[0]):
    if YtestTrue[k, 0] == 1:
        countPreIctal += 1
        if yhat_keras[k, 0] == 1:
            goodPreIctal += 1 
        elif yhat_keras[k,1] == 1:
            PreIctalAsIctal += 1
        else:
            PreIctalAsHealthy += 1

    elif YtestTrue[k, 1] ==  1:
        countIctal += 1
        if yhat_keras[k,1] == 1:
            goodIctal += 1
        elif yhat_keras[k,0] == 1:
            IctalAsPreIctal += 1
        else:
            IctalAsHealthy +=1

    elif YtestTrue[k, 2] ==  1:
        countHealthy +=1
        if yhat_keras[k,2] == 1:
            goodHealthy +=1
        elif yhat_keras[k,0] == 1:
            HealthyAsPreIctal +=1
        else:
            HealthyAsIctal +=1
            
            
TestSF.write(" Confusion Matrix\n")
TestSF.write("             | Inter-Ictal | Pre-Ictal | Ictal\n")
TestSF.write("----------------------------------------------\n")
TestSF.write(" Inter-Ictal |     {}      |    {}     |  {}  \n".format(goodHealthy, PreIctalAsHealthy, IctalAsHealthy))
TestSF.write("----------------------------------------------\n") 
TestSF.write(" Pre-Ictal   |     {}      |    {}     |  {}  \n".format(HealthyAsPreIctal, goodPreIctal, IctalAsPreIctal))
TestSF.write("----------------------------------------------\n") 
TestSF.write(" Ictal       |     {}      |    {}     |  {}  \n".format(HealthyAsIctal, PreIctalAsHealthy, goodIctal))
TestSF.write("\n")
TestSF.write("Correctly classified inter-ictals: {}/{} = {}%\n".format(goodHealthy, countHealthy, goodHealthy/countHealthy*100))
TestSF.write("Correctly classified pre-ictals  : {}/{} = {}%\n".format(goodPreIctal, countPreIctal, goodPreIctal/countPreIctal*100))
TestSF.write("Correctly classified ictals      : {}/{} = {}%\n".format(goodIctal, countIctal, goodIctal/countIctal*100))
TestSF.write("\n")
TestSF.write(" \n")




##################
Wconv_trained = (model1.layers[0].get_weights()[0]).reshape(2,2,2)
Wconv_flat = Wconv_trained.reshape(4,2)
Bconv_trained = (model1.layers[0].get_weights()[1]).reshape(1,2)

WGRU = np.asarray(model1.layers[3].get_weights()[0])
UGRU = np.asarray(model1.layers[3].get_weights()[1])
BGRU = np.asarray(model1.layers[3].get_weights()[2]).reshape(1, -1)

Wlin_trained = np.asarray(model1.layers[4].get_weights()[0])
Blin_trained = np.asarray(model1.layers[4].get_weights()[1]).reshape(1,-1)

def save_param(filename, array):
    np.savetxt(filename, array, delimiter=",")
    return

Wz_trained, Wr_trained, Wh_trained = WGRU[:,0:100], WGRU[:,100:200], WGRU[:,200:300]
Uz_trained, Ur_trained, Uh_trained = UGRU[:,0:100], UGRU[:,100:200], UGRU[:,200:300]
Bz_trained, Br_trained, Bh_trained = BGRU[:,0:100], BGRU[:,100:200], BGRU[:,200:300]
TestSF.write("Wconv\n")
save_param(TestSF, Wconv_flat)
TestSF.write("Bconv\n")
save_param(TestSF, Bconv_trained)
TestSF.write("Wz\n")
save_param(TestSF, Wz_trained)
TestSF.write("Wr\n")
save_param(TestSF, Wr_trained)
TestSF.write("Wh\n")
save_param(TestSF, Wh_trained)
TestSF.write("Uz\n")
save_param(TestSF, Uz_trained)
TestSF.write("Ur\n")
save_param(TestSF, Ur_trained)
TestSF.write("Uh\n")
save_param(TestSF, Uh_trained)
TestSF.write("Bz\n")
save_param(TestSF, Bz_trained)
TestSF.write("Br\n")
save_param(TestSF, Br_trained)
TestSF.write("Bh\n")
save_param(TestSF, Bh_trained)
TestSF.write("Wlin\n")
save_param(TestSF, Wlin_trained)
TestSF.write("Blin\n")
save_param(TestSF, Blin_trained)

TestSF.close()