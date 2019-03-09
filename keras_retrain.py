# Used to retrain  a full-precision net in lower precision

import numpy as np
import keras
import re
from random import randint
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GRU, TimeDistributed, Flatten, Softmax, Dropout, MaxPooling1D, Add, Input
from keras.activations import relu, softmax
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from random import randint
import target_data_gen_100seq_GENERAL
#from target_data_gen import get_sizes
#from target_data_gen import target_gen
from target_data_gen_100seq_GENERAL import batching_dataset_01, get_sizes_train, target_gen
#temp
from target_data_gen_100seq_GENERAL import get_sizes_test
from imblearn.over_sampling import SMOTE, ADASYN


from model_BNN import QuantizedDense, QuantizedGRU, QuantizedConv2D

# ***************************************#
def standardization(data, variance=1, avg=0):
    mean = np.mean(data)+avg
    var = np.var(data)
    data = variance*(data-mean)/var
    return data

def get_minmax(data):
    maxx = np.amax(data)
    minn = np.amin(data)
    return minn, maxx

def normalization(data, minn, maxx):
    data = data/maxx
    #data = abs(data)/maxx
    return data

def class_occurence(labels):
    nb_healthy = np.sum(labels[:,2])
    nb_pre_ictal = np.sum(labels[:,0])
    nb_ictal = np.sum(labels[:,1])
    return nb_healthy, nb_pre_ictal, nb_ictal

def numpy_quantize(X):
    m = 1.0
    f = 4.0
    Xclip = np.where(X > 2**m, 2**m*np.ones_like(X), X)
    Xclip = np.where(Xclip < -2**m, -2**m*np.ones_like(Xclip), Xclip)
    Xround = np.round(2**f*Xclip)*2**(-f)
    return Xround

print("Initializing ...")
np.random.seed(5)
output = 3
#X=np.array(sigbufs
startfile = 9
batch_size = 100 #(length)
batch_mod = 10000
timesteps = 10
seq_number = 100
patient = 1
dataset_size = 26
initial_file = 1
electrodes = 23
inputs = 23
X, Y = list(), list()


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
#**********************CALLLING DATA GENERATOR FUNCTION ***********************

X, inputs, length = get_sizes_train(X, dataset_size, initial_file, patient) #Defining sizes for input/target data

Y=np.zeros((seq_number*dataset_size, output))

#***********************CALLING TARGET GENERATOR FUNCTION**********************

file = '../database/chb01-summary.txt'
if seq_number == 50 and batch_size == 400:
    Y = target_gen(output, dataset_size, batch_size, seq_number, timesteps, file)
    file_targets = open("../database/chb01-targets_50seq.txt", 'w')
if seq_number == 100 and batch_size == 50:
    Y = target_gen(output, dataset_size, batch_size, seq_number, timesteps, file)
    file_targets = open("../database/chb01-targets_50batch.txt", 'w')
if seq_number == 100 and batch_size == 100:
    Y = target_gen(output, dataset_size, batch_size, seq_number, timesteps, file)
    file_targets = open("../database/chb01-targets", 'w')
np.savetxt(file_targets, Y,  delimiter = ",")
file_targets.close()
#******************************************************************************

print("Reading edf files ...")


dataset_size = 18 #Re-defining dataset size for training
#Defining sizes for input/target data
X_new=np.zeros((seq_number*dataset_size, timesteps, inputs, batch_size, 1))
Y_new= Y[(initial_file-1)*seq_number:dataset_size*seq_number]
for i in range(dataset_size):
    print("Ynew{}".format(i+1), Y_new[i*seq_number,:])
print(Y_new.shape," Y_new shape")
for m in range(0,dataset_size):
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
            X_new[seq_number*m+i,j,:,0:batch_size,0] =  X[m,0,0,0:inputs,initial:final,0]
    

print( "Normalizing data ...")            


# Instantiate the cross validator
print("Ynew shape", Y_new.shape)
nb_healthy, nb_pre_ictal, nb_ictal = class_occurence(Y_new)
print("Class occurence: Ictal {}, PreIctal {}, Healthy {}".format(nb_ictal, nb_pre_ictal, nb_healthy))
minn, maxx = get_minmax(X_new)
xtrain = normalization(X_new, minn, maxx)
ytrain = Y_new


xtrain = xtrain.reshape(-1,electrodes * batch_size * timesteps)
#X_resampled, Y_resampled = ADASYN().fit_resample(xtrain, ytrain)
#X_resampled, Y_resampled = SMOTE().fit_resample(xtrain, ytrain)
X_resampled, Y_resampled = xtrain, ytrain
xtrain = xtrain.reshape(-1, timesteps, electrodes, batch_size,1)
X_resampled = X_resampled.reshape(-1, timesteps, electrodes, batch_size,1)

#----------------------------------------------GET PRETRAINED WEIGHTS---------------------------------------------#
Wconv_flat, Bconv, Wz, Wr, Wh, Uz, Ur, Uh, Bz, Br, Bh, Wlin, Blin = [],[],[],[],[],[],[],[],[],[],[],[],[]

BoolWconv, BoolBconv,BoolWz, BoolWr, BoolWh, BoolUz, BoolUr, BoolUh, BoolBz, BoolBr, BoolBh, BoolWlin, BoolBlin = False, False, False, False, False, False, False, False, False, False, False, False, False
with open("../results_low_prec/keras_param_3class_30e.txt") as file_Wall:
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

#------------------------------------MODEL CREATION----------------------------------------#

model = Sequential()
#quantize inputs & conv weights  ---> done inside QuantizedConv2D
model.add(TimeDistributed(QuantizedConv2D(2,(2,2), activation = 'relu'), input_shape = (None,electrodes,batch_size,1)))
model.layers[0].set_weights((Wconv.reshape(2,2,1,2),Bconv.reshape(2,)))
#quantize activations ---> No Need, reLU is not "unQuantizing".
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
#do nothing
model.add(TimeDistributed(Flatten()))
#quantize input, weights, recurrent weights ---> done in Quantized GRU
model.add(QuantizedGRU(100, recurrent_dropout = 0.3, dropout = 0.3))
model.layers[3].set_weights((np.concatenate((Wz, Wr, Wh), axis = 1), np.concatenate((Uz, Ur, Uh), axis = 1), np.concatenate((Bz.reshape(100,), Br.reshape(100,), Bh.reshape(100,)), axis = 0)))
model.add(Dropout(0.0))
#need to Quantize the weights and input here.
model.add(QuantizedDense(output, activation  = 'softmax'))
model.layers[5].set_weights((Wlin, Blin.reshape(3,)))
adam = Adam(lr =0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 10.0**-7, decay = 0.0, amsgrad = False)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

#------------------------------------TRAINING PHASE----------------------------------------#

model.fit(X_resampled, Y_resampled, epochs = 30, shuffle = True)


#------------------------------------WEIGHTS SAVING----------------------------------------#
    
Wconv_trained = numpy_quantize(model.layers[0].get_weights()[0])
Wconv_flat = Wconv_trained.reshape(4,2)
Bconv_trained = numpy_quantize(model.layers[0].get_weights()[1])

WGRU = numpy_quantize(np.asarray(model.layers[3].get_weights()[0]))
UGRU = numpy_quantize(np.asarray(model.layers[3].get_weights()[1]))
BGRU = numpy_quantize(np.asarray(model.layers[3].get_weights()[2]))

Wlin_trained = numpy_quantize(np.asarray(model.layers[5].get_weights()[0]))
Blin_trained = numpy_quantize(np.asarray(model.layers[5].get_weights()[1]))


Wz_trained, Wr_trained, Wh_trained = WGRU[:,0:100], WGRU[:,100:200], WGRU[:,200:300]
Uz_trained, Ur_trained, Uh_trained = UGRU[:,0:100], UGRU[:,100:200], UGRU[:,200:300]
Bz_trained, Br_trained, Bh_trained = BGRU[0:100], BGRU[100:200], BGRU[200:300]

#------------------------------------TEST PHASE------------------------------------------#


def save_param(filename, array):
    np.savetxt(filename, array, delimiter=",")
    return

startfile = 18
nb_files = 8

# temporary code
X_test=np.zeros((seq_number*nb_files, timesteps, inputs, batch_size, 1))


for file_iter in range(startfile, nb_files+startfile):
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
      
            X_test[(m-startfile)*seq_number+i,j,:,0:batch_size,0] =  X[m,0,0,0:inputs,initial:final,0]
print("Normalizing data ... ")

X_test = normalization(X_test, minn, maxx)
print("maxx", maxx)
 
print("Predicting ... {}")
yhat = model.predict(X_test, verbose=0)
file2 = open("../results_low_prec/test_retrain19to26.txt", 'w')
np.savetxt(file2, yhat, delimiter="," )
file2.close()
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
Y_stat = np.argmax(yhat, axis = 1)
yhat = keras.utils.to_categorical(Y_stat, 3)
for k in range(YtestTrue.shape[0]):
    if YtestTrue[k, 0] == 1:
        countPreIctal += 1
        if yhat[k, 0] == 1:
            goodPreIctal += 1 
        elif yhat[k,1] == 1:
            PreIctalAsIctal += 1
        else:
            PreIctalAsHealthy += 1

    elif YtestTrue[k, 1] ==  1:
        countIctal += 1
        if yhat[k,1] == 1:
            goodIctal += 1
        elif yhat[k,0] == 1:
            IctalAsPreIctal += 1
        else:
            IctalAsHealthy +=1

    elif YtestTrue[k, 2] ==  1:
        countHealthy +=1
        if yhat[k,2] == 1:
            goodHealthy +=1
        elif yhat[k,0] == 1:
            HealthyAsPreIctal +=1
        else:
            HealthyAsIctal +=1

TestSF = open("../results_low_prec/test_retrain19to26.txt", 'a')
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
TestSF.close()

file_Wall = open("../results_low_prec/keras_param_3class_30e_5bits_onlysign.txt", 'w')
file_Wall.write("Wconv\n")
save_param(file_Wall, Wconv_flat)
file_Wall.write("Bconv\n")
save_param(file_Wall, Bconv_trained)
file_Wall.write("Wz\n")
save_param(file_Wall, Wz_trained)
file_Wall.write("Wr\n")
save_param(file_Wall, Wr_trained)
file_Wall.write("Wh\n")
save_param(file_Wall, Wh_trained)
file_Wall.write("Uz\n")
save_param(file_Wall, Uz_trained)
file_Wall.write("Ur\n")
save_param(file_Wall, Ur_trained)
file_Wall.write("Uh\n")
save_param(file_Wall, Uh_trained)
file_Wall.write("Bz\n")
save_param(file_Wall, Bz_trained)
file_Wall.write("Br\n")
save_param(file_Wall, Br_trained)
file_Wall.write("Bh\n")
save_param(file_Wall, Bh_trained)
file_Wall.write("Wlin\n")
save_param(file_Wall, Wlin_trained)
file_Wall.write("Blin\n")
save_param(file_Wall, Blin_trained)
file_Wall.close()