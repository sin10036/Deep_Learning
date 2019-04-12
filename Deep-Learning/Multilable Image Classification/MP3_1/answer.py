import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3,16, 5)
        self.batchnorm1=nn.BatchNorm2d(16)
        

        self.conv2 = nn.Conv2d(16,20, 3,padding=1)
        self.batchnorm2=nn.BatchNorm2d(20)
        
        
        self.conv3 = nn.Conv2d(20,16,3,padding=1)
        self.batchnorm3=nn.BatchNorm2d(16)

        
        self.conv4 = nn.Conv2d(16,8,3,padding=1)
        self.batchnorm4=nn.BatchNorm2d(8)
        
        self.conv5 = nn.Conv2d(8,16,3,padding=1)
        self.batchnorm5=nn.BatchNorm2d(16)
      
        self.pool1 = nn.MaxPool2d(3,2)

        self.conv6 = nn.Conv2d(16,10, 3)
        self.batchnorm6=nn.BatchNorm2d(10)

        self.pool2 = nn.MaxPool2d(3,2)
        
        
        self.conv7 = nn.Conv2d(10,5, 3,padding=1)
        self.batchnorm7=nn.BatchNorm2d(5)
        self.pool3 = nn.MaxPool2d(2,2)
        

        self.conv8 = nn.Conv2d(5,10, 3,padding=1)
        self.batchnorm8=nn.BatchNorm2d(10)
        self.pool8 = nn.MaxPool2d(3,2)
        
        
        self.conv9 = nn.Conv2d(10,10, 3,padding=1)
        self.batchnorm9=nn.BatchNorm2d(10)
        self.pool9 = nn.MaxPool2d(2,1)
        
        
        self.fc1 = nn.Linear(10 *12 *12, 300)
        self.fc2 = nn.Linear(300, NUM_CLASSES)
        #self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = (F.relu(self.batchnorm1(self.conv1(x))))

        residual1=x
        
       
        x = (F.relu(self.batchnorm2(self.conv2(x))))
        
        
        a=self.batchnorm3(self.conv3(x))
        
        
        x = F.relu(a+(residual1))
        
        residual2=x

        x = (F.relu(self.batchnorm4(self.conv4(x))))
        x = self.pool1(F.relu((self.batchnorm5(self.conv5(x)))+(residual2)))
        
        x = self.pool2(F.relu(self.batchnorm6(self.conv6(x))))
        
        
        x = self.pool3(F.relu(self.batchnorm7(self.conv7(x))))


        x = self.pool8(F.relu(self.batchnorm8(self.conv8(x))))
        
        x = self.pool9(F.relu(self.batchnorm9(self.conv9(x))))
        
         
        
        x = x.view(x.size()[0], 12 * 12 * 10)
        x = F.relu(self.fc1(x))
        #x=self.dropout(x)
        #x = F.relu(self.fc2(x))
       # x=self.dropout1(x)
        x = self.fc2(x)
        return x