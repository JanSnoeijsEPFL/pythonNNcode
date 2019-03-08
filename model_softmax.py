import numpy as np
from random import randint


#******************************************** INITIALIZERS ********************************************************#

def orthogonal_initializer(shape, lim):
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)
    a = np.random.uniform(-lim,lim, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q[:shape[0], :shape[1]]

#*************************************************** GRU ***********************************************************#

class GRU():
    #Wz, Wr, Wh, Uz, Ur, Uh = np.array()
    #z, r, h, s = np.array()
    
    def __init__(self,sequences, timesteps, inputs, outputs):
        print("creating GRU layer")
        lim_w = np.sqrt(6/(inputs+outputs))
        lim_u = np.sqrt(6/(2*outputs))
        lim_wlin = np.sqrt(6/(outputs+3))
        self.Wz = np.random.uniform(-lim_w,lim_w,(inputs, outputs))
        self.Wr = np.random.uniform(-lim_w,lim_w,(inputs, outputs))
        self.Wh = np.random.uniform(-lim_w,lim_w,(inputs, outputs))
        self.Bz = np.zeros((1, outputs))
        self.Br = np.zeros((1, outputs))
        self.Bh = np.zeros((1, outputs))
        self.Wlin = np.random.uniform(-lim_wlin, lim_wlin, (outputs, 3))
        self.Blin = np.zeros((1,3))
        print(self.Wz.shape)
        self.y, self.dy = np.zeros((sequences, 3), dtype = np.float64) ,np.zeros((sequences,3))
        #self.Uz, self.Ur, self.Uh = np.random.uniform(-lim_u,lim_u,(outputs, outputs)), np.random.uniform(-lim_u,lim_u,(outputs, outputs)), np.random.uniform(-lim_u,lim_u,(outputs, outputs))
        self.Uz = orthogonal_initializer((outputs, outputs), lim_u)
        self.Ur = orthogonal_initializer((outputs, outputs), lim_u)
        self.Uh = orthogonal_initializer((outputs, outputs), lim_u)
        self.z, self.r = np.zeros((sequences,timesteps,outputs), dtype =np.float64),np.zeros((sequences,timesteps,outputs), dtype=np.float64)
        self.h, self.s = np.zeros((sequences,timesteps,outputs),dtype =np.float64),np.zeros((sequences,timesteps,outputs), dtype = np.float64)
        print(self.s.shape)
        self.dWlin = np.zeros((outputs,1))
        self.GRU_updateBz = Optimizer(self.Bz)
        self.GRU_updateBr = Optimizer(self.Br)
        self.GRU_updateBh = Optimizer(self.Bh)
        self.GRU_updateWz = Optimizer(self.Wz)
        self.GRU_updateWr = Optimizer(self.Wr)
        self.GRU_updateWh = Optimizer(self.Wh)
        self.GRU_updateUz = Optimizer(self.Uz)
        self.GRU_updateUr = Optimizer(self.Ur)
        self.GRU_updateUh = Optimizer(self.Uh)
        self.Lin_updateW = Optimizer(self.Wlin)
        self.Lin_updateB = Optimizer(self.Blin)
        #self.s_aug = np.zeros((sequences, timesteps, outputs+1))
        #self.s_aug[:,:,0].fill(1) # bias
    def forward(self,X):
        # initialize
        # first iteration
        #print(self.z.shape, X.shape)
        outputs = self.s.shape[2]
        self.z[:,0,:] = hard_sigmoid(np.matmul(X[:,0,:],self.Wz) + self.Bz) #[seq * time * OUT] = [seq * time * IN] @ [IN * OUT]
        self.r[:,0,:] = hard_sigmoid(np.matmul(X[:,0,:],self.Wr) + self.Br)
        self.h[:,0,:] = tanh(np.matmul(X[:,0,:],self.Wh) + self.Bh)
        self.s[:,0,:] = (1-self.z[:,0,:])*self.h[:,0,:]
        for t in range(1, X.shape[1]):
            self.z[:,t,:] = hard_sigmoid(np.matmul(X[:,t,:],self.Wz) + np.matmul(self.s[:,t-1,:],self.Uz) + self.Bz) #[samples,outputs]+[1, outputs]
            self.r[:,t,:] = hard_sigmoid(np.matmul(X[:,t,:],self.Wr) + np.matmul(self.s[:,t-1,:],self.Ur) + self.Br)
            self.h[:,t,:] = tanh(np.matmul(X[:,t,:],self.Wh) + np.matmul((self.r[:,t,:] *self.s[:,t-1,:]),self.Uh) + self.Bh)
            self.s[:,t,:] = self.z[:,t,:]*self.s[:,t-1,:] + (1-self.z[:,t,:])*self.h[:,t,:]
            #self.s_aug [:,t,1:outputs+1] = self.s[:,t,:]
        self.y = softmax(np.matmul(self.s[:,self.s.shape[1]-1,:],self.Wlin) + self.Blin)
        return self.y
    
    def backward(self,dy, X):
        inputs = X.shape[2]
        outputs = self.s.shape[2]
        dsnext = np.zeros_like(self.s[:,0,:])
        dsnext = np.matmul(dy,np.transpose(self.Wlin))
        dX = np.zeros((X.shape))
        dWlin = np.matmul(np.transpose(self.s[:,self.s.shape[1]-1,:]),dy)
        dBlin = np.sum(dy, axis = 0).reshape(1,-1)
        dWz, dWr, dWh = np.zeros((inputs, outputs)), np.zeros((inputs, outputs)), np.zeros((inputs, outputs))
        dUz, dUr, dUh = np.zeros((outputs, outputs)), np.zeros((outputs, outputs)), np.zeros((outputs, outputs))
        dBz, dBr, dBh = np.zeros((1, outputs)), np.zeros((1, outputs)), np.zeros((1, outputs))
        for t in reversed(range(X.shape[1])):
            #print(ds.shape, self.z.shape)
            
            ds = dsnext
            dh = ds*(1-self.z[:,t,:])
            dh_l = dh*tanh(self.h[:,t,:], deriv=True)
           
            dWh += np.matmul(np.transpose(X[:,t,:]),dh_l) 
            if t != 0:
                dUh += np.matmul(np.transpose(self.r[:,t,:]*self.s[:,t-1,:]),dh_l)
            dBh += np.sum(dh_l, axis = 0).reshape(1,-1)
            
            drsp = np.matmul(dh_l,np.transpose(self.Uh))
            if t != 0:
                dr = drsp * self.s[:,t-1,:]
                dr_l = dr * hard_sigmoid(np.matmul(X[:,t,:],self.Wr) + np.matmul(self.s[:,t-1,:],self.Ur) + self.Br, deriv=True) 
                dWr += np.matmul(np.transpose(X[:,t,:]),dr_l) 
                dUr += np.matmul(np.transpose(self.s[:,t-1,:]),dr_l) 
                dBr += np.sum(dr_l, axis = 0).reshape(1,-1)
            else:
                dr = np.zeros_like(self.s[:,t-1,:])
                dr_l = np.zeros_like(dr)
            
            if t != 0:
                dz = (self.s[:,t-1,:]-self.h[:,t,:]) * ds  
                dz_l = dz * hard_sigmoid(np.matmul(X[:,t,:],self.Wz) + np.matmul(self.s[:,t-1,:],self.Uz) + self.Bz, deriv=True)
            else:
                dz = -self.h[:,t,:]*ds
                dz_l = dz * hard_sigmoid(np.matmul(X[:,t,:],self.Wz) + self.Bz, deriv=True)
                
            dWz += np.matmul(np.transpose(X[:,t,:]),dz_l)
            if t != 0:
                dUz += np.matmul(np.transpose(self.s[:,t-1,:]),dz_l)
            dBz += np.sum(dz_l, axis = 0).reshape(1,-1)
            
            # calculate gradient w.r.t s[t-1]
            ds_fz_inner = np.matmul(dz_l,np.transpose(self.Uh))
            ds_fz = ds * self.z[:,t,:]
            ds_fh = drsp * self.r[:,t,:] 
            ds_fr = np.matmul(dr_l,np.transpose(self.Ur))
            
            dsnext = ds_fz_inner + ds_fz + ds_fh + ds_fr
            
            for n in reversed(range(t)):
                dX[:,n,:]+=np.matmul(dh_l,np.transpose(self.Wh)) + np.matmul(dr_l,np.transpose(self.Wr)) + np.matmul(dz_l,np.transpose(self.Wz))
            
            #update weights
        self.Wz = self.GRU_updateWz.adam_update(self.Wz, dWz)
        self.Wr = self.GRU_updateWr.adam_update(self.Wr, dWr)
        self.Wh = self.GRU_updateWh.adam_update(self.Wh, dWh)
        self.Uz = self.GRU_updateUz.adam_update(self.Uz, dUz)
        self.Ur = self.GRU_updateUr.adam_update(self.Ur, dUr)
        self.Uh = self.GRU_updateUh.adam_update(self.Uh, dUh) 
        self.Bz = self.GRU_updateBz.adam_update(self.Bz, dBz)
        self.Br = self.GRU_updateBr.adam_update(self.Br, dBr)
        self.Bh = self.GRU_updateBh.adam_update(self.Bh, dBh)
        self.Wlin = self.Lin_updateW.adam_update(self.Wlin, dWlin)
        self.Blin = self.Lin_updateB.adam_update(self.Blin, dBlin)
        return ds, dX
    
    def change_input_size(self, sequences, timesteps,outputs):
        self.z, self.r = np.zeros((sequences,timesteps,outputs),dtype =np.float16),np.zeros((sequences,timesteps,outputs),dtype =np.float16)
        self.h, self.s = np.zeros((sequences,timesteps,outputs),dtype =np.float16),np.zeros((sequences,timesteps,outputs),dtype =np.float16)
        self.s_aug = np.zeros((sequences, timesteps, outputs+1),dtype =np.float16)
        self.y = np.zeros((sequences, timesteps, 3),dtype =np.float16)
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
        self.W = np.random.uniform(-1,1,(kernel_height, kernel_width, filters))
        self.B = np.zeros((1, filters))
        self.dW = np.zeros((kernel_height, kernel_width, filters))
        self.dB = np.zeros((1, filters))
        print('Creating Conv2D layer')
        self.Conv2D_updateW = Optimizer(self.W)
        self.Conv2D_updateB = Optimizer(self.B)
    # W.shape : [kernel_height, kernel_width, nb_filters]
    # X.shape : [samples, timesteps, height, width, 1]
    # B.shape : [1, nb_filters]
    def forward(self, X):
        #print("conv2d Xshape", X.shape)
        self.height = X.shape[2]
        self.width = X.shape[3]
        self.M = self.W.shape[0]
        self.N = self.W.shape[1]
        self.K = self.W.shape[2]
        self.nb_seq = X.shape[0]
        self.timesteps = X.shape[1]
        #print(self.timesteps)
        #compute new dimensions
        self.new_height = self.height - self.M + 1
        self.new_width = self.width - self.N + 1
        
       # print(new_height, new_width)
        h = np.zeros((self.nb_seq, self.timesteps, self.new_height, self.new_width, self.K))
        for k in range(self.K):
            for i in range(self.new_height):
                for j in range(self.new_width):
                    h[:,:,i,j,k]=np.sum(X[:,:,i:i+self.M, j:j+self.N,0]*self.W[:,:,k], axis =(2,3))+self.B[0,k]
        file_verify_conv = open("verify_conv2D_custom.txt", 'w')
        np.savetxt(file_verify_conv, self.W.reshape(4,2))
        np.savetxt(file_verify_conv, self.B)
       # np.savetxt(file_verify_conv, h.reshape(-1,))
        file_verify_conv.close()
        return h
    # dH has dimensions of H which means in case of X.shape=[3,3,1], W.shape=[2,2,1] => H.shape=[2,2,1]
    # dX should have the same shape as X, i.e dX.shape=[3,3,1] = dH conv2D flipped W 'FULL'
    def backward(self, dH, X):
        # dw is the same operation as in forward propagation. 
        # no need to compute dX in our case because the conv2D layer is the first layer and means the end of backprop. algorithm.
        for k in range(self.K):
            self.dB[:,k] = np.sum(dH[:,:,:,:,k])
            for i in range(self.M):
                for j in range(self.N):
                    #average over all the sequences and timesteps
                    self.dW[i,j,k]=np.sum(X[:,:,i:i+self.new_height, j:j+self.new_width,0]*dH[:,:,:,:,k])
        self.W = self.Conv2D_updateW.adam_update(self.W, self.dW)
        self.B = self.Conv2D_updateB.adam_update(self.B, self.dB)
        return
    
    def return_param(self):
        return self.W, self.B
    
    def set_param(self, W, B):
        self.W = W
        self.B = B
        return
class fast_conv2D:
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
        self.W = np.random.uniform(-1,1,(kernel_height, kernel_width, filters))
        self.B = np.zeros((1, filters))
        self.dW = np.zeros((kernel_height, kernel_width, filters))
        self.dB = np.zeros((1, filters))
        print('Creating Conv2D layer')
        self.Conv2D_updateW = Optimizer(self.W)
        self.Conv2D_updateB = Optimizer(self.B)
    # W.shape : [kernel_height, kernel_width, nb_filters]
    # X.shape : [samples, timesteps, height, width, 1]
    # B.shape : [1, nb_filters]
    def forward(self, X):
        #print("conv2d Xshape", X.shape)
        self.height = X.shape[2]
        self.width = X.shape[3]
        self.M = self.W.shape[0]
        self.N = self.W.shape[1]
        self.K = self.W.shape[2]
        self.nb_seq = X.shape[0]
        self.timesteps = X.shape[1]
        #print(self.timesteps)
        #compute new dimensions
        self.new_height = self.height - self.M + 1
        self.new_width = self.width - self.N + 1
        
       # print(new_height, new_width)
       # h = np.zeros((self.nb_seq, self.timesteps, self.new_height, self.new_width, self.K))
        #for k in range(self.K):
       #     for i in range(self.new_height):
        #        for j in range(self.new_width):
        #            h[:,:,i,j,k]=np.sum(X[:,:,i:i+self.M, j:j+self.N,0]*self.W[:,:,k], axis =(2,3))+self.B[0,k]
        x_reshaped=X[:,:,:self.new_height*self.M, :self.new_width*self.N,:].reshape(self.nb_seq, self.nb_timesteps, self.new_height, self.M, self.new_width, self.N, -1)
        
        x_reshaped=x_reshaped.reshape(self.nb_seq, self.nb_timesteps, self.new_height, self.new_width, -1,self.N*self.M)
        
        W = self.W.swapaxes(0,2)
        W = W.reshape(self.K, -1)
        file_verify_conv = open("verify_conv2D_custom.txt", 'w')
        np.savetxt(file_verify_conv, self.W.reshape(4,2))
        np.savetxt(file_verify_conv, self.B)
        np.savetxt(file_verify_conv, h.reshape(-1,))
        file_verify_conv.close()
        return h
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

        return H, X


    def backward(self, X_argmax, dH):
        dX = np.zeros((X_argmax.shape),dtype =np.float16)
        for k in range(self.K):
            for i in range(0, self.height, self.M):
                for j in range(0,self.width, self.N):
                    dX[:,:, i:i+self.M,j:j+self.N,k] = self.norm_argmax(X_argmax[:,:,i:i+self.M, j:j+self.N, k], dH[:,:,int(i/self.M), int(j/self.N), k])       
        return dX
                                                                
    def norm_argmax(self, X, dH):
        #print("X_argmax shape maxpool = ", X.shape)
        h = X.shape[2]
        w = X.shape[3]
        I = np.array(np.arange(X.shape[2]+X.shape[3]))
        Ibig = np.zeros_like(X)
        #print("Maxpoolbackward:", X.shape, Ibig.shape, I.shape)
        Ibig = np.tile(I,(X.shape[0], X.shape[1],1))
        #print("ibig", Ibig.shape)
        Xresh = X.reshape(X.shape[0], X.shape[1], Ibig.shape[2]) # multidimensional
        diff = (Xresh-Ibig).astype(int)
        #duplicate dH x 4
        dH_dupl = np.zeros_like(X)
        for m in range(2):
            for n in range(2):
                dH_dupl[:,:,m,n] = dH
        # reshape to make boolean assignment
        Xresh = Xresh.flatten()
        diff = diff.flatten()
        dHflat = dH_dupl.flatten()
        #print("dH shape", dH.shape)
        Xresh[diff != 0] = -1
        #print("xresh shape", Xresh.shape, diff.shape)
        deriv = np.copy(Xresh)
        deriv[Xresh > -1.0] = dHflat[Xresh > -1.0]
        deriv[Xresh == -1.0] = 0.0
        deriv = deriv.reshape(X.shape[0],X.shape[1],h, w)
        #print("deriv shape max pool = ", deriv.shape)
        return deriv

    #
    
#-----------------------------------
class fast_MaxPool2D():
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
        #if self.height%2 == 1:
        #    X = np.delete(X, obj=self.height-1, axis=2)
        #    self.height = X.shape[2]
        if self.width%2 == 1:
            X = np.delete(X, obj=self.width-1, axis=3)
            self.width = X.shape[3]
        X_argmax = np.copy(X)
      
        new_height = int(self.height/self.M)
        new_width = int(self.width/self.N)
      
        H =X[:,:,:new_height*self.M, :new_width*self.N,:].reshape(self.nb_seq, self.nb_timesteps, new_height, self.M, new_width, self.N, -1).max(axis=(3, 5))
        
        return H, X_argmax
    
#********************************************* ACTIVATIONS ****************************************************#


def hard_sigmoid(input, deriv = False): #requires input of forward prop for derivative.
    if deriv:
        temp = 0.2*np.maximum(-2.5, np.minimum(2.5, input))
        temp[input > 2.5] = 0.0
        temp[input < -2.5] = 0.0
        return temp
    else:
        return np.maximum(0, np.minimum(1, (0.2*input + 0.5)))
def tanh(input, deriv=False): #requires output of forward prop for derivative
    if deriv:
        return 1 - input ** 2
    else:
        return np.tanh(input)
    
def hard_tanh(input, deriv=False):
    if deriv:
        a = input.shape
        temp = np.ones_like(input)
        temp[input > 1] = 0.0
        temp[input < -1] = 0.0
    else:
        return np.maximum(-1, np.minimum(1, (input)))
def softmax(input): # input is > 2D

    exps = np.exp(input - np.max(input))
    res = np.zeros_like(exps)
    for i in range(exps.shape[1]):
        res[:,i] = np.divide(exps[:,i], np.sum(exps, axis=1))
    return res

def reLU(input, deriv=False): #requires input of forward prop for derivative
    if deriv:
       
        output = np.maximum(0, input)
        temp = np.copy(output)
        output[temp != 0 ]= 1.0
        return output
    else:
        output = np.maximum(0, input)
        return output

#****************************************** LOSS FUNCTIONS ********************************************************#

#loss function modified to Multiclass
def CrossEntropy(yHat, y):
    output = np.sum(-np.log(yHat)*y, axis=1)
    output1 = np.sum(output)/800
    return output1, output

    
#***************************************** OPTIMIZERS ************************************************************#

#weight update
class Optimizer():
    # initialize training parameters
    def __init__(self, W):
       # self.Wconv = np.zeros((nb_Wconv1, nb_Wconv2))
       # self.Bconv = np.zeros((nb_Bconv))
       # self.Wr, self.Wh, self.Wz = np.zeros((nb_GRUin, nb_GRUout)), np.zeros((nb_GRUin, nb_GRUout)), np.zeros((nb_GRUin, nb_GRUout))
       # self.Ur, self.Uh, self.Uz = np.zeros((nb_GRUout, nb_GRUout)), np.zeros((nb_GRUout, nb_GRUout)), np.zeros((nb_GRUout, nb_GRUout))
       # self.Wlin = np.zeros((nb_Wlin)) # only 1 output
        self.alpha = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 10**-7
        self.v = np.zeros((W.shape))
        self.m = np.zeros((W.shape))
        self.n_iters = 1
    def adam_update(self, W, dW):
        self.m = self.beta1*self.m + (1 - self.beta1)*dW
        self.v = self.beta2*self.v + (1 - self.beta2)*np.power(dW, 2)
        m_corr = self.m/(1-self.beta1)
        v_corr = self.v/(1-self.beta2)
        W = W  - self.alpha*m_corr/(np.sqrt(v_corr)+self.eps)
        #W = pow2_ternarization(W)
        return W

#*************************************** Quantization *******************************************************#

def pow2_ternarization(W):
    nbits = 16
    nbfrac = 13
    nbint = nbits-nbfrac
    shape = W.shape
    Wbin = np.minimum(2**nbint,np.maximum(-2**nbint, W))
    Wbin = np.round(Wbin*2**nbfrac)*2**(-nbfrac)
    return Wbin


