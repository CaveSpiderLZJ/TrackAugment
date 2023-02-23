import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cf
from train.feature import feature


class Model(nn.Module):
    
    
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.feature_extractor = feature
        feature_channels = 24
        window_length = int(cf.WINDOW_DURATION * cf.FS_TRAIN)
        self.conv1 = nn.Conv1d(in_channels=feature_channels, out_channels=20, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=window_length, out_channels=20, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1)
        self.conv6 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1)
        self.fc1 = nn.Linear(400, 32)
        self.fc2 = nn.Linear(32, cf.N_CLASSES)
        
    
    def forward(self, x) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = F.relu(self.conv1(x)).transpose(1, 2)
        x = F.relu(self.conv2(x)).transpose(1, 2)
        x = (x + F.relu(self.conv3(x))).transpose(1, 2)
        x = (x + F.relu(self.conv4(x))).transpose(1, 2)
        x = (x + F.relu(self.conv5(x))).transpose(1, 2)
        x = x + F.relu(self.conv6(x))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
if __name__ == '__main__':
    m = nn.Conv1d(in_channels=24, out_channels=20, kernel_size=1)
    input = torch.randn(10, 24, 300)
    output = m(input)
    print(output.shape)
        