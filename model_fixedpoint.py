import numpy as np
from random import randint
from rig.type_casts import \
        NumpyFloatToFixConverter, NumpyFixToFloatConverter

s14 = NumpyFloatToFixConverter(signed=True, n_bits=8, n_frac=4)
f4 = NumpyFixToFloatConverter(4)
f8 = NumpyFixToFloatConverter(8)
f = 4

def Quantize(X):
    m = 1.0
    f = 4.0
    Xclip = np.where(X > 2**m, 2**m*np.ones_like(X), X)
    Xclip = np.where(Xclip < -2**m, -2**m*np.ones_like(Xclip), Xclip)
    Xround = np.round(2**f*Xclip)*2**(-f)
    return Xround
#*************************************************** GRU ***********************************************************#

class GRU():
    #Wz, Wr, Wh, Uz, Ur, Uh = np.array()
    #z, r, h, s = np.array()
    
    def __init__(self,sequences, timesteps, inputs, outputs):
        print("creating GRU layer")
        self.Wz = np.zeros((inputs, outputs), dtype = np.float16)
        self.Wr = np.zeros((inputs, outputs), dtype = np.float16)
        self.Wh = np.zeros((inputs, outputs), dtype = np.float16)
        self.Bz = np.zeros((1, outputs), dtype = np.float16)
        self.Br = np.zeros((1, outputs), dtype = np.float16)
        self.Bh = np.zeros((1, outputs), dtype = np.float16)
        self.Wlin = np.zeros((outputs, 3), dtype = np.float16)
        self.Blin = np.zeros((1,3), dtype = np.float16)
        self.y = np.zeros((sequences, 3), dtype = np.float16)
        self.y_16 = np.zeros((sequences, 3), dtype = np.int16)
        self.Uz = np.zeros((outputs, outputs), dtype = np.float16)
        self.Ur = np.zeros((outputs, outputs), dtype = np.float16)
        self.Uh = np.zeros((outputs, outputs), dtype = np.float16)
        self.z, self.r = np.zeros((sequences,timesteps,outputs), dtype = np.float16),np.zeros((sequences,timesteps,outputs), dtype = np.float16)
        self.h, self.s = np.zeros((sequences,timesteps,outputs),dtype = np.float16),np.zeros((sequences,timesteps,outputs), dtype = np.float16)
        self.z_16, self.r_16 = np.zeros((sequences,timesteps,outputs), dtype = np.float16),np.zeros((sequences,timesteps,outputs), dtype = np.float16)
        self.h_16, self.s_16 = np.zeros((sequences,timesteps,outputs),dtype = np.float16),np.zeros((sequences,timesteps,outputs), dtype = np.float16)
    def forward(self,X):
        # initialize
        # first iteration
        #print(self.z.shape, X.shape)
        outputs = self.s.shape[2]
        self.z_16[:,0,:] = np.matmul(X[:,0,:],self.Wz) + self.Bz #[seq * time * OUT] = [seq * time * IN] @ [IN * OUT]
        self.z[:,0,:] = hard_sigmoid(self.z_16[:,0,:])
        self.r_16[:,0,:] = np.matmul(X[:,0,:],self.Wr) + self.Br
        self.r[:,0,:] = hard_sigmoid(self.r_16[:,0,:])
        self.h_16[:,0,:] = np.matmul(X[:,0,:],self.Wh)+ self.Bh
        self.h[:,0,:] = hard_tanh(self.h_16[:,0,:])
        self.s_16[:,0,:] = (1-self.z[:,0,:])*self.h[:,0,:]
        self.s[:,0,:] = Quantize(self.s_16[:,0,:])
        temp = np.zeros_like(self.s_16[:,0,:])
        for t in range(1, X.shape[1]):
            self.z_16[:,t,:] = np.matmul(X[:,t,:],self.Wz) + np.matmul(self.s[:,t-1,:],self.Uz) + self.Bz #[samples,outputs]+[1, outputs]
            self.z[:,t,:] = hard_sigmoid(self.z_16[:,t,:])
            self.r_16[:,t,:] = np.matmul(X[:,t,:],self.Wr) + np.matmul(self.s[:,t-1,:],self.Ur) + self.Br
            self.r[:,t,:] = hard_sigmoid(self.r_16[:,t,:])
            temp = self.r[:,t,:] *self.s[:,t-1,:]
            temp = Quantize(temp)
            self.h_16[:,t,:] = np.matmul(X[:,t,:],self.Wh) + np.matmul(temp,self.Uh) + self.Bh
            self.h[:,t,:] = hard_tanh(self.h_16[:,t,:])
            self.s_16[:,t,:] = self.z[:,t,:]*self.s[:,t-1,:] + (1-self.z[:,t,:])*self.h[:,t,:]
            self.s[:,t,:] = Quantize(self.s_16[:,t,:])
        self.y_16 = np.matmul(self.s[:,self.s.shape[1]-1,:],self.Wlin) + self.Blin
        self.y = softmax(self.y_16)
        return self.y
    
    def return_param(self):
        return self.Wz, self.Wr, self.Wh, self.Uz, self.Ur, self.Uh, self.Bz, self.Br, self.Bh, self.Wlin, self.Blin
    
    def set_param(self, Wz, Wr, Wh, Uz, Ur, Uh, Bz, Br, Bh, Wlin, Blin):
        self.Wz = Wz
        self.Wr = Wr
        self.Wh = Wh
        self.Uz = Uz
        self.Ur = Ur
        self.Uh = Uh
        self.Bz = Bz
        self.Br = Br
        self.Bh = Bh
        self.Wlin = Wlin
        self.Blin = Blin
        return
    
#********************************************************** CONV2D **************************************************#

class Conv2D():
    height = 0
    width = 0
    nb_seq = 0
    timesteps = 0
    new_height = 0
    new_width = 0
    K = 0
    M = 0
    N = 0
    def __init__(self, kernel_height, kernel_width, filters):
        self.W = np.zeros((kernel_height, kernel_width, filters), dtype = np.float16)
        self.B = np.zeros((1, filters), dtype = np.float16)
        print('Creating Conv2D layer')
   
    # W.shape : [kernel_height, kernel_width, nb_filters]
    # X.shape : [samples, timesteps, height, width, 1]
    # B.shape : [1, nb_filters]
    def forward(self, X):
    
        self.height = X.shape[2]
        self.width = X.shape[3]
        self.M = self.W.shape[0]
        self.N = self.W.shape[1]
        self.K = self.W.shape[2]
        self.nb_seq = X.shape[0]
        self.timesteps = X.shape[1]
    
        #compute new dimensions
        self.new_height = self.height - self.M + 1
        self.new_width = self.width - self.N + 1
    
        h = np.zeros((self.nb_seq, self.timesteps, self.new_height, self.new_width, self.K), dtype = np.float16)
        h_32 = np.zeros((self.nb_seq, self.timesteps, self.new_height, self.new_width, self.K), dtype = np.float32)
        for k in range(self.K):
            for i in range(self.new_height):
                for j in range(self.new_width):
                    #h[:,:,i,j,k]=np.sum(X[:,:,i:i+self.M, j:j+self.N,0]*self.W[:,:,k]*2**-f, axis =(2,3))+self.B[0,k]
                    h_32[:,:,i,j,k]=np.sum(X[:,:,i:i+self.M, j:j+self.N,0]*self.W[:,:,k], axis =(2,3))+self.B[0,k]
        h = Quantize(h_32)
        return h
    
    def return_param(self):
        return self.W, self.B
    
    def set_param(self, W, B):
        self.W = W
        self.B = B
        return

#*************************************** MAX POOL 2D *******************************************************#

class MaxPool2D():
    height = 0
    width = 0
    M = 0
    N = 0
    K = 0
    nb_seq = 0
    timesteps = 0
    def __init__(self):
        print('creating 2D Max pooling layer')
    
    def forward(self, X, pool):
        self.height = X.shape[2]
        self.width = X.shape[3]
        self.M = pool[0]
        self.N = pool[1]
        self.K = X.shape[4]
        self.nb_seq = X.shape[0]
        self.nb_timesteps = X.shape[1]
        #X_argmax = np.copy(X)

        new_height = self.height//self.M
        new_width = self.width//self.N

        H =X[:,:,:new_height*self.M, :new_width*self.N,:].reshape(self.nb_seq, self.nb_timesteps, new_height, self.M, new_width, self.N, -1).max(axis=(3, 5))

        return H
#********************************************* ACTIVATIONS ****************************************************#


def hard_sigmoid(input): #requires input of forward prop for derivative.
    return Quantize(np.maximum(0, np.minimum(1, (0.2*input + 0.5))))

def hard_tanh(input):
    return Quantize(np.maximum(-1, np.minimum(1, (input))))

def softmax(input): # input is > 2D
    input = Quantize(input)
    exps = np.exp(input - np.max(input))
    res = np.zeros_like(exps)
    for i in range(exps.shape[1]):
        res[:,i] = np.divide(exps[:,i], np.sum(exps, axis=1))
    return res

def reLU(input): #requires input of forward prop for derivative
    output = np.maximum(0, input)
    return output

