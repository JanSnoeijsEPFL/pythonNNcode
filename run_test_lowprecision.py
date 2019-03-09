# imports

import numpy as np
import re
from random import randint

from rig.type_casts import \
        NumpyFloatToFixConverter, NumpyFixToFloatConverter
import model_fixedpoint
from model_fixedpoint import GRU, Conv2D, MaxPool2D, reLU

# ***************************************#

print("Preparing for test phase ...")

maxx = 1061.00122101221
seq_number = 100
nb_files = 8
timesteps = 10
inputs = 23
batch_size = 100
startfile = 18
minn = 0
electrodes = 23
X = list()
patient = 1
dataset_size = 26
minn = 0

conv_filters = 2
conv_height = 2
conv_width = 2
GRUoutputs = 100
GRUinputs = int((inputs-1)/2)*int((batch_size-1)/2)*conv_filters

def Quantize(X):
    m = 1.0
    f = 4.0
    Xclip = np.where(X > 2**m, 2**m*np.ones_like(X), X)
    Xclip = np.where(Xclip < -2**m, -2**m*np.ones_like(Xclip), Xclip)
    Xround = np.round(2**f*Xclip)*2**(-f)
    return Xround

def normalization(data, minn, maxx):
    data = data/maxx
    #data = abs(data)/maxx
    return data

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

# to be transformed to simulate a real-time case:
    # store the pre-read X_test file into a text file (not normalized and in full precision, otherwise it would be cheating!)
    # write a function to read 1 sequence at a time from Xtest.txt (2D array --> 1 line = 1 sequence, after 23x100 numbers --> new timestep, after 100 numbers new electrode etc...)
    # make the algorithm run/per 1 sequence and evaluate the delay !
    
X_test=np.zeros((seq_number*nb_files, timesteps, inputs, batch_size, 1), dtype=np.float16)



print("Normalizing data ... ")
file_test = open("../database/RT_datastream.txt", 'r')
X_test = np.loadtxt(file_test, delimiter = ",")
file_test.close()

X_test = normalization(X_test, minn, maxx)
X_test = Quantize(X_test)
#convert to numpy uint8 array :)
print("maxx", maxx)
X_test = X_test.reshape(seq_number*nb_files, timesteps, batch_size, electrodes, 1)
#---------------------------------------------------------------------------------------------------------------------------#
Wconv_flat, Bconv, Wz, Wr, Wh, Uz, Ur, Uh, Bz, Br, Bh, Wlin, Blin = [],[],[],[],[],[],[],[],[],[],[],[],[]
        
#load from textfiles
BoolWconv, BoolBconv,BoolWz, BoolWr, BoolWh, BoolUz, BoolUr, BoolUh, BoolBz, BoolBr, BoolBh, BoolWlin, BoolBlin = False, False, False, False, False, False, False, False, False, False, False, False, False
with open("../results_low_prec/keras_param_3class_30e_5bits_onlysign.txt") as file_Wall:
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

Wconv_flat = np.asarray(Wconv_flat, dtype = np.float16)
Bconv = np.asarray(Bconv, dtype = np.float16)
Wz = np.asarray(Wz, dtype = np.float16)
Wr = np.asarray(Wr, dtype = np.float16)
Wh = np.asarray(Wh, dtype = np.float16)
Uz = np.asarray(Uz, dtype = np.float16)
Ur = np.asarray(Ur, dtype = np.float16)
Uh = np.asarray(Uh, dtype = np.float16)
Bz = np.asarray(Bz, dtype = np.float16)
Br = np.asarray(Br, dtype = np.float16)
Bh = np.asarray(Bh, dtype = np.float16)
Wlin = np.asarray(Wlin, dtype = np.float16)
Blin = np.asarray(Blin, dtype = np.float16)
Wconv = Wconv_flat.reshape(2,2,2)


#--------------------------------MODEL CREATION-----------------------------------#

layer_0 = Conv2D(conv_height,conv_width,conv_filters)
# add activation
layer_1 = MaxPool2D()
layer_2 = GRU(seq_number*nb_files, timesteps, GRUinputs, GRUoutputs)


layer_0.set_param(Wconv, Bconv.reshape(1,2))
layer_2.set_param(Wz, Wr, Wh, Uz, Ur, Uh, Bz.reshape(1,100), Br.reshape(1,100), Bh.reshape(1,100), Wlin, Blin.reshape(1,3))

#----------------------------------------------------------------------------------#

print("Generating predictions ...")
HConv = layer_0.forward(X_test) # 5D data for train_data, 3D for Wconv 2D for Bconv
store_conv = open("../results_low_prec/fpconvfloat.txt", 'w')
np.savetxt(store_conv, HConv[0,:,:,:,:].reshape(-1))
store_conv.close()
YConv = reLU(HConv) # no requirement on shape
#pooling layer
pool_kernel = np.array([2,2])
YPool = layer_1.forward(YConv,  pool_kernel) #5D data for YConv
#flattening
X_GRU_flat = YPool.reshape(YPool.shape[0],10,-1) # check size here should be 3D (100*nb_files, 10, 1078)

#GRU
yhat_test = Quantize(layer_2.forward(X_GRU_flat)) # timesteps
file2 = open("../results_low_prec/custom4fracbits.txt", 'w')
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
yhat_test[Y_stat == 0,0] = 1
yhat_test[Y_stat == 0,1] = 0
yhat_test[Y_stat == 0,1] = 0

yhat_test[Y_stat == 1,1] = 1
yhat_test[Y_stat == 1,0] = 0
yhat_test[Y_stat == 1,2] = 0

yhat_test[Y_stat == 2,2] = 1
yhat_test[Y_stat == 2,0] = 0
yhat_test[Y_stat == 2,1] = 0

#yhat_test = keras.utils.to_categorical(Y_stat, 3)
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

TestSF = open("../results_low_prec/custom4fracbits.txt", 'a')
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
TestSF.write("\n")
TestSF.close()
