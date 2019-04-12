import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import copy

NUM_CLASSES = 21

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 20, 3, padding=1)
        self.conv3 = nn.Conv2d(20, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv5 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(8, 5, 3, padding=1)
        self.conv7 = nn.Conv2d(5, 10, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(5)
        self.bn5 = nn.BatchNorm2d(10)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, NUM_CLASSES)

    def forward(self, x): 
        x = F.relu(self.bn2(self.conv1(x)))

        y = x
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(F.relu(self.bn2(self.conv3(x)) + y)) #residual

        y = x
        x = F.relu(self.bn3(self.conv4(x)))
        x = self.pool(F.relu(self.bn2(self.conv5(x)) + y)) #residual

        x = F.relu(self.bn3(self.conv4(x)))
        x = self.pool(F.relu(self.bn4(self.conv6(x))))
        x = self.pool(F.relu(self.bn5(self.conv7(x))))

        x = x.view(x.size()[0], 10 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x












