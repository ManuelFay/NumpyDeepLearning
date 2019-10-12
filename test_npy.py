import numpy as np

import framework_npy as nn


def compute_nb_errors(model,criterion,train_input,train_target,test_input, test_target, mini_batch_size,eta,epochs=25, printV = True):


    model = nn.train_model(model,criterion,train_input,train_target,mini_batch_size,eta,epochs,printV=printV)

    #Train accuracy
    output = np.zeros((len(train_input),1))
    for i in range(train_input.shape[0]):
        tmp = model(np.expand_dims(train_input[i],1)).reshape(1,-1)
        output[i,:] = tmp
       
       
    
    if nb_out == 1:
        accuracy = (((((output.squeeze()>0).astype('float')*2)-1)-train_target)==0).sum()/train_target.shape[0]
    else:
        accuracy = ((output.squeeze().argmax(dim=1) - train_target.argmax(dim=1))==0).sum()/train_input.shape[0]
    print('Train Accuracy: {}\n'.format(accuracy))


    #Test accuracy
    output = np.zeros((len(test_input),1))
    for i in range(test_input.shape[0]):
        tmp = model(np.expand_dims(test_input[i],1)).reshape(1,-1)
        output[i,:] = tmp
        
    if nb_out == 1:
        accuracy = ((((output.squeeze()>0).astype('float')*2-1)-test_target)==0).sum()/test_target.shape[0]
    else:
        accuracy = ((output.squeeze().argmax(dim=1) - test_target.argmax(dim=1))==0).sum()/test_input.shape[0]
        
    return accuracy, output



#Declaring the model class
class SimpleNet(nn.Sequencer):
        
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(nb_in,200)
        self.fc5 = nn.Linear(200,nb_out)
        #self.tan1 = nn.Tanh()
        self.sig1 = nn.Sigmoid()

        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        #self.seq = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3, self.relu3, self.fc4, self.relu4,self.fc5]
        self.seq = [self.fc1, self.relu1, self.fc5, self.sig1]
        
        for func in self.seq:
            x = func(x)
        return x


#Declaring the model class
class DemandedNet(nn.Sequencer):
        
    def __init__(self):
        super(DemandedNet, self).__init__()
        self.fc1 = nn.Linear(nb_in,25)
        self.fc2 = nn.Linear(25,25)
        self.fc3 = nn.Linear(25,25)
        self.fc5 = nn.Linear(25,nb_out)
        #self.sig1 = nn.Sigmoid()
        self.tan1 = nn.Tanh()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        
        
    def forward(self, x):
        self.seq = [self.fc1, self.relu1, self.fc2,  self.fc3,self.fc5,self.tan1]
        #self.seq = [self.fc1, self.relu1, self.fc5, self.tan1]
        
        for func in self.seq:
            x = func(x)
        return x


# # Generate test set

def generate_disc_set(nb):
    inputs = np.random.uniform(-1,1,(nb,2))
    targets = (np.sqrt(inputs[:,0]**2 + inputs[:,1]**2)<0.79788456)

    mu,std = inputs.mean(0),inputs.std(0)
    inputs = (inputs-mu)/std
    return inputs,(targets*2)-1



print('\n One class')

#Creating the data
nb_out = 1
nb_in  = 2

nb = 1000
split = 0.8

inputs, targets = generate_disc_set(nb)

while(abs(targets.sum().item()-500) <10):
    inputs, targets = generate_disc_set(nb)

train_input = inputs[:int(split*nb),:]
test_input = inputs[int(split*nb):,:]
train_target = targets[:int(split*nb)]
test_target = targets[int(split*nb):]


criterion = nn.MSELoss()
eta, mini_batch_size = 1e-4, 100

model = DemandedNet() #SimpleNet

accuracy, output = compute_nb_errors(model,criterion,train_input,train_target,test_input,test_target,mini_batch_size,eta,epochs=50) 
print('Test Accuracy: {}'.format(accuracy))
