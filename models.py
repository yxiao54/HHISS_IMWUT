import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.distributions import Normal



   
class Baseline(nn.Module):
    def __init__(self,hidden=64,drop=0.2,input_dim=384,num_class=4):
        super(Baseline, self).__init__()
        
        dropout=drop
        self.flat=nn.Flatten()

        self.fc1=nn.Linear(input_dim,hidden)
        self.drop=nn.Dropout(p=dropout)
        self.relu=nn.ReLU()
        
     
        self.fc2=nn.Linear(hidden,hidden)
        
  
        self.fc3=nn.Linear(hidden,hidden)
     
        self.fc4=nn.Linear(hidden,num_class)

        self.softmax=nn.Softmax(-1)

        self.loss_class = nn.CrossEntropyLoss()
        

        
    def forward(self, x): # input: batch_size x 6 x 800
        #x=self.flat(x)
        #x=self.bn1(x)
        x=self.fc1(x)
        x=self.drop(x)
        x=self.relu(x)
        
        #x=self.bn2(x)
        x=self.fc2(x)
        x=self.drop(x)
        x=self.relu(x)
        
        #x=self.bn3(x)
        x=self.fc3(x)
        x=self.drop(x)
        x=self.relu(x)
        
        #x=self.bn4(x)
        x=self.fc4(x)
        x=self.drop(x)
        x=self.softmax(x)


        
        return x
    def draw(self, x): # input: batch_size x 6 x 800
        #x=self.flat(x)
        #x=self.bn1(x)
        x=self.fc1(x)
        x=self.drop(x)
        x=self.relu(x)
        
        #x=self.bn2(x)
        x=self.fc2(x)
        x=self.drop(x)
        x=self.relu(x)
        
        #x=self.bn3(x)
        x=self.fc3(x)
        x=self.drop(x)
        x=self.relu(x)
        emb=x
        #x=self.bn4(x)
        x=self.fc4(x)
        x=self.drop(x)
        x=self.softmax(x)


        
        return x,emb
