#######################################################################################################
###						CustomNN.py		
###	  Written by Ethan Cannaert, adapted from 					    
###  https://debuggercafe.com/american-sign-language-recognition-using-deep-learning/
###		
###	  Inherits from the nn.Module class and defines 4 conv, 2 FC, 
###	  6 MaxPool2d, and 1 adaptive_avg_pool2d layers and sets out the order of these in forward().
###	  
#######################################################################################################

import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
### get binarized labels to define shape of output tensor layer 
binarized_labels_pkl = open(os.path.join('%s'%os.sep.join(os.path.dirname(__file__).split(os.sep)[:-2] ))+'/processedDatasets/labels_binarized.pickle', 'rb')
binarized_labels_loaded = pickle.load(binarized_labels_pkl)[0]

### custon CNN, inherits from base torch.nn 
class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN,self).__init__()  ## override __init__ to that of the nn.Module base class
        self.conv1 = nn.Conv2d(3,16,5) # 3 channels in, 16 channels out, kernel size 5
        self.conv2 = nn.Conv2d(16,32,5)
        self.conv3 = nn.Conv2d(32,64,3)
        self.conv4 = nn.Conv2d(64,128,5)
        
        self.fc1 = nn.Linear(128,256)
        self.fc2 = nn.Linear(256, len(binarized_labels_loaded))
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self,x): # passed in input tensor
        x = self.pool(F.relu(self.conv1(x))) # convolutional layer followed by relu mapping and then a pool layer
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        ### get the tensor shape
        t_layers, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x,1).reshape(t_layers,-1)    ### reshape this to be 1D, dimensions are inferred 
        x = F.relu(self.fc1(x)) 
        return self.fc2(x) ### return tensor of length 29 