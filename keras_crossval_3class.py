# imports

#produces a text file with set weights from training phase.
#Organization of file:

# first layer Conv2D: (2,2,2) (2)
# second layer GRU  : 1078 inputs, 100 units -> (1078,100,3) (100,100,3) (100,3)
# final layer linear: (100,1) (1)
# TOTAL: 353811 values to save (Save to Fixed point format after stochastic rounding?)

import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GRU, TimeDistributed, Flatten, Softmax, Dropout, MaxPooling1D, Add, Input, GlobalAveragePooling1D
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

from sklearn.model_selection import StratifiedKFold


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
nb_train_files = 18
electrodes = 23
X, Y = list(), list()

#**********************CALLLING DATA GENERATOR FUNCTION ***********************

X, input_size, length = get_sizes_train(X, dataset_size, initial_file, patient) #Defining sizes for input/target data

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

input_size_new = 23
batch_size_new = 100
dataset_size = 18 #Re-defining dataset size for training
#Defining sizes for input/target data
X_new=np.zeros((seq_number*dataset_size, timesteps, input_size_new, batch_size_new, 1))
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
            initial = initial_time+(i*batch_size_new*timesteps)+j*batch_size_new
            final = initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new)
            #print(initial_time+(i*batch_size_new*timesteps)+j*batch_size_new)
            #print(initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new))
            X_new[seq_number*m+i,j,:,0:batch_size_new,0] =  X[m,0,0,0:input_size_new,initial:final,0]
    

print( "Normalizing data ...")            


kfold_splits = 5
# Instantiate the cross validator
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)
print("Ynew shape", Y_new.shape)
nb_healthy, nb_pre_ictal, nb_ictal = class_occurence(Y_new)
print("Class occurence: Ictal {}, PreIctal {}, Healthy {}".format(nb_ictal, nb_pre_ictal, nb_healthy))
minn, maxx = get_minmax(X_new)
X_new = normalization(X_new, minn, maxx)
Y_1D = np.argmax(Y_new, axis = 1)

CVSF = open("../results_3class_fullKeras/cross_val_res_30e_noUPSAMPLING_dropoutPreGRU04.txt", 'w')
CVSF.write("Cross validation results 30 epochs + noUPSAMPLING\n")

for index, (train_indices, val_indices) in enumerate(skf.split(X_new, Y_1D)):
    Y_new = keras.utils.to_categorical(Y_1D, num_classes = 3)
    nb_healthy, nb_pre_ictal, nb_ictal = class_occurence(Y_new)
    print("Class occurence: Ictal {}, PreIctal {}, Healthy {}".format(nb_ictal, nb_pre_ictal, nb_healthy))

    print("Training on fold " + str(index+1) + "/{}...".format(kfold_splits))
    xtrain, xval = X_new[train_indices], X_new[val_indices]
    ytrain, yval = Y_new[train_indices], Y_new[val_indices]

    
    #----------------------------------------------------------
    xtrain = xtrain.reshape(-1,electrodes * batch_size * timesteps)
    #X_resampled, Y_resampled = ADASYN().fit_resample(xtrain, ytrain)
    #X_resampled, Y_resampled = SMOTE().fit_resample(xtrain, ytrain)
    X_resampled, Y_resampled = xtrain, ytrain
    xtrain = xtrain.reshape(-1, timesteps, electrodes, batch_size,1)
    X_resampled = X_resampled.reshape(-1, timesteps, electrodes, batch_size,1)
    #----------------------------------------------------------------------------

    nb_healthy, nb_pre_ictal, nb_ictal = class_occurence(Y_resampled)
    print("Class occurence: Ictal {}, PreIctal {}, Healthy {}".format(nb_ictal, nb_pre_ictal, nb_healthy))

    #***************** TRAINING PHASE ******************************


    model = Sequential()
    model.add(TimeDistributed(Conv2D(2,(2,2), activation = 'relu'), input_shape = (None,electrodes,batch_size,1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))

    model.add(GRU(100, recurrent_dropout = 0.3, dropout = 0.3, return_sequences = True))
    model.add(Dropout(0.0))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(output, activation  = 'softmax'))

    adam = Adam(lr =0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 10.0**-7, decay = 0.0, amsgrad = False)
    model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(model.summary())


    model.fit(X_resampled, Y_resampled, epochs = 60, shuffle = True)
    Y_cross_val = model.predict(xval)
    
    Y_stat = np.argmax(Y_cross_val, axis = 1)
    Y_cross_val = keras.utils.to_categorical(Y_stat, 3)
    Y_true = np.argmax(yval, axis = 1) 
    
    goodPreIctal = 0
    goodIctal = 0
    goodHealthy = 0
    countPreIctal =0
    countIctal = 0
    countHealthy = 0
    IctalAsPreIctal = 0
    IctalAsHealthy = 0
    PreIctalAsHealthy = 0
    PreIctalAsIctal = 0
    HealthyAsPreIctal = 0
    HealthyAsIctal = 0
    for k in range(yval.shape[0]):
        if yval[k, 0] == 1:
            countPreIctal +=1
            if Y_cross_val[k, 0] == 1:
                goodPreIctal +=1
            elif Y_cross_val[k,1] == 1:
                PreIctalAsIctal +=1
            else:
                PreIctalAsHealthy +=1
                
        elif yval[k, 1] ==  1:
            countIctal +=1
            if Y_cross_val[k,1] == 1:
                goodIctal +=1
            elif Y_cross_val[k,0] == 1:
                IctalAsPreIctal +=1
            else:
                IctalAsHealthy +=1
                
        elif yval[k, 2] ==  1:
            countHealthy +=1
            if Y_cross_val[k,2] == 1:
                goodHealthy +=1
            elif Y_cross_val[k,0] == 1:
                HealthyAsPreIctal +=1
            else:
                HealthyAsIctal +=1
  
    print(" well-classified pre-ictals : {} out of {}".format(goodPreIctal, countPreIctal))
    print(" well-classified ictals : {} out of {}".format(goodIctal, countIctal))
    print(" well-classified inter-ictal : {} out of {}".format(goodHealthy, countHealthy))
    print(" --------------------------------------------")
    print(" Ictals classified as Pre-ictals: {}".format(IctalAsPreIctal))
    print(" Ictals classified as Inter-ictals: {}".format(IctalAsHealthy))
    print(" --------------------------------------------")
    print(" Pre-ictals classified as Inter-ictals: {}".format(PreIctalAsHealthy))
    print(" Pre-ictals classified as Ictals: {}".format(PreIctalAsIctal))
    print(" --------------------------------------------")
    print(" Inter-ictals classified as Pre-ictals: {}".format(HealthyAsPreIctal))
    print(" Inter-ictals classified as Ictals: {}".format(HealthyAsIctal))


   
    CVSF.write("Fold {}\n".format(index+1))
    CVSF.write(" Confusion Matrix\n")
    CVSF.write("             | Inter-Ictal | Pre-Ictal | Ictal\n")
    CVSF.write("----------------------------------------------\n")
    CVSF.write(" Inter-Ictal |     {}      |    {}     |  {}  \n".format(goodHealthy, PreIctalAsHealthy, IctalAsHealthy))
    CVSF.write("----------------------------------------------\n") 
    CVSF.write(" Pre-Ictal   |     {}      |    {}     |  {}  \n".format(HealthyAsPreIctal, goodPreIctal, IctalAsPreIctal))
    CVSF.write("----------------------------------------------\n") 
    CVSF.write(" Ictal       |     {}      |    {}     |  {}  \n".format(HealthyAsIctal, PreIctalAsHealthy, goodIctal))
    CVSF.write("\n")
    CVSF.write("Correctly classified inter-ictals: {}/{} = {}%\n".format(goodHealthy, countHealthy, goodHealthy/countHealthy*100))
    CVSF.write("Correctly classified pre-ictals  : {}/{} = {}%\n".format(goodPreIctal, countPreIctal, goodPreIctal/countPreIctal*100))
    CVSF.write("Correctly classified ictals      : {}/{} = {}%\n".format(goodIctal, countIctal, goodIctal/countIctal*100))
    CVSF.write("\n")
    CVSF.write(" \n")

CVSF.close()
    
Wconv_trained = (model.layers[0].get_weights()[0])
Wconv_flat = Wconv_trained.reshape(4,2)
Bconv_trained = (model.layers[0].get_weights()[1])

WGRU = np.asarray(model.layers[3].get_weights()[0])
UGRU = np.asarray(model.layers[3].get_weights()[1])
BGRU = np.asarray(model.layers[3].get_weights()[2])

Wlin_trained = np.asarray(model.layers[6].get_weights()[0])
Blin_trained = np.asarray(model.layers[6].get_weights()[1])


Wz_trained, Wr_trained, Wh_trained = WGRU[:,0:100], WGRU[:,100:200], WGRU[:,200:300]
Uz_trained, Ur_trained, Uh_trained = UGRU[:,0:100], UGRU[:,100:200], UGRU[:,200:300]
Bz_trained, Br_trained, Bh_trained = BGRU[0:100], BGRU[100:200], BGRU[200:300]




def save_param(filename, array):
    np.savetxt(filename, array, delimiter=",")
    return
inputs = 23
startfile = 18
nb_files = 8
seq_number = 100
batch_size = 100
timesteps = 10


# test phase


# temporary code
X_test=np.zeros((seq_number*nb_files, timesteps, inputs, batch_size, 1))
patient = 1

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
 
print("Predicting ... {}".format(file_iter+1))
yhat = model.predict(X_test, verbose=0)
file2 = open("../results_3class_fullKeras/test_19to26_30ep_noUPSAMPLING_dropoutPreGRU04.txt", 'w')
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

TestSF = open("../results_3class_fullKeras/test_19to26_30ep_noUPSAMPLING_dropoutPreGRU04.txt", 'a')
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

file_Wall = open("keras_param_3class_30e_stateful.txt", 'w')
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