import numpy as np 

def sigmoid(x, derivative=False):
    ret =  x*(1-x) if derivative else np.power((1+np.exp(-x)),-1)
    #print(x.shape,ret.shape)
    return ret


class Module(object):
    
    def __call__(self,x):
        return self.forward(x)
    
    def forward(self,*input):
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        raise NotImplementedError
        
    def backward(self , *gradwrtoutput):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect 
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        raise NotImplementedError
        
    def zero_grad(self):
        return 'ok'
    
    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient  
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return [x for x in vars(self).keys()]        


# In[4]:


class Linear(Module):
    
    def __init__(self, sizeInput, sizeOutput): 
        self.sizeInput  = sizeInput
        self.sizeOutput = sizeOutput
        #eps = 1e-6
        #eps = 0.1
        eps = 1/(sizeInput**.5)
        self.w = np.random.normal(0,eps,(sizeInput,sizeOutput))
        self.b = np.random.normal(0,eps,(sizeOutput,1))
        self.input = None
        self.wgrad = np.zeros((sizeInput,sizeOutput))
        self.bgrad = np.zeros((sizeOutput,1))
        
        
    def forward(self,x):
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        self.input = x
        return np.matmul(x.reshape(1,-1),self.w).reshape(-1,1) + self.b
        
    def backward(self , gradwrtoutput):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect 
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        gradwrtinput = np.matmul(self.w,gradwrtoutput)
        
        self.bgrad += gradwrtoutput
        self.wgrad += np.matmul(gradwrtoutput,self.input.reshape(1,-1)).transpose()
        #self.wgrad += np.matmul(gradwrtoutput.reshape(-1, 1),self.input.reshape(1,-1)).t()
    
        return gradwrtinput
    
    def zero_grad(self):
        self.wgrad = np.zeros((self.sizeInput,self.sizeOutput))
        self.bgrad = np.zeros((self.sizeOutput,1))
     
    def sub_grad(self,eta):
        self.w = self.w - eta*self.wgrad
        self.b = self.b - eta*self.bgrad

    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient  
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return [[self.w, self.wgrad],[self.b,self.bgrad]]     


# In[5]:


class Tanh(Module):
    
    def __init__(self): 
        #maybe initialize a tensor to keep in memory the inputs?
        self.input = None
        
    def forward(self,x):
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        self.input = x
        return np.tanh(x)
        
    def backward(self , x):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect 
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input.""" 
        #return (self.input.cosh().power(-2))*x
        return (4 * (np.exp(self.input) + np.exp(self.input*(-1)))**(-2))*x
    
    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient  
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return []     

class Sigmoid(Module):
    
    def __init__(self): 
        #maybe initialize a tensor to keep in memory the inputs?
        self.input = None
        
    def forward(self,x):
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        self.input = x
        return sigmoid(x)
        
    def backward(self , x):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect 
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input.""" 
        #return (self.input.cosh().power(-2))*x
        return sigmoid(x,derivative=True)
    
    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient  
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return []  

# In[6]:


class ReLU(Module):
    
    def __init__(self): 
        #maybe initialize a tensor to keep in memory the inputs?
        self.input = None
        
    def forward(self,x):
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        self.input = x
        return (x >= 0).astype('float')*x
        
    def backward(self , x):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect 
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        # 1 if >0, 0 otherwise
        return ((self.input >= 0).astype('float'))*x
    
    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient  
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return []     


# In[7]:


class MSELoss(Module):
    
    def __init__(self): 
        self.t = None
        self.v = None
        
    def __call__(self,v,t):
        return self.forward(v,t)
        
    def forward(self,v,t):
        #loss function
        self.t = t
        self.v = v
        #print(t.shape,v.shape)
        return ((t-v)**2).sum()
        
    def backward(self):
        #dloss
        #d((t1-v1)^2 + (t2-v2)^2 +(t3-v3)^2 + ...)/dvi = d/dvi (ti^2 - 2tivi + vi^2) = 2(vi-ti)

        return 2*(self.v-self.t)

class BCELoss(Module):
    
    def __init__(self): 
        self.t = None
        self.v = None
        
    def __call__(self,v,t):
        return self.forward(v,t)
        
    def forward(self,v,t):
        #loss function
        self.v = v
        self.t = t

        return (-(t*np.log(sigmoid(v))+(1-t)*np.log(1-sigmoid(v))))
        
    def backward(self):
        #dloss
        #d((t1-v1)^2 + (t2-v2)^2 +(t3-v3)^2 + ...)/dvi = d/dvi (ti^2 - 2tivi + vi^2) = 2(vi-ti)

        return self.t*(sigmoid(self.v)-1)+(1-self.t)*sigmoid(self.v)



def train_model(model,criterion,train_input,train_target,mini_batch_size,eta,epochs=25, printV = False):
    for e in range(0, epochs):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.shape[0], mini_batch_size):
            model.zero_grad()
            for i in range(mini_batch_size):
                output = model(np.expand_dims(train_input[i+b],1))
                loss = criterion(output.squeeze(1), np.expand_dims(train_target[i+b],1))
                sum_loss = sum_loss + loss.item()
                model.backward(criterion.backward())
                
            for p in model.paramFunc():
                p.sub_grad(eta)
                
        if printV == True:
            print("Epoch: {} \t -> Loss: {} ".format(e, sum_loss))
            
    return model


class Sequencer(Module):
    def __init__(self):
        self.seq = None
        
    def add(self,inseq):
        self.seq=inseq
    
    def forward(self, x):
        
        for func in self.seq:
            x = func(x)
        return x
      
    def backward(self, lossBack):
        x = lossBack
        for func in self.seq[::-1]:
            x = func.backward(x)
        
    def param(self):
        params = []
        for func in self.seq:
            for p in func.param():
                params.append(p)
        return params

    def paramFunc(self):
        params = []
        for func in self.seq:
            if len(func.param())>0:
                params.append(func)
        return params
    
    def zero_grad(self):
        if self.seq:
            for func in self.seq:
                func.zero_grad()
    
