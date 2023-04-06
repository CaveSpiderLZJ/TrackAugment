import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cf
from train.feature import *


class ConvBnAct(nn.Module):
    
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int,
            stride:int=1, padding:int=0, groups:int=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
    
class SEBlock(nn.Module):
    
    
    def __init__(self, channels, reduction):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.SiLU(), nn.Conv1d(channels // reduction, channels, kernel_size=1), nn.Sigmoid())
        
        
    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class MBConv(nn.Module):
    
    
    def __init__(self, in_channels, out_channels, expansion, reduction, stride:int=1):
        super().__init__()
        self.shortcut = (in_channels == out_channels and stride == 1)
        expanded_channels = in_channels * expansion
        self.expand = ConvBnAct(in_channels, expanded_channels,
            kernel_size=3, stride=stride, padding=1, groups=1)
        self.se = SEBlock(expanded_channels, reduction)
        self.compress = ConvBnAct(expanded_channels, out_channels, 1)
    
    
    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.se(x)
        x = self.compress(x)
        if self.shortcut:
            x += residual
        return x
    

class SeparableConv(nn.Module):
    
    
    def __init__(self, in_channels:int, out_channels:int,
            kernel_size:int, stride:int=1, padding:int=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                groups=in_channels, stride=stride, padding=padding)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
    
class LinearBnDropout(nn.Module):
    
    
    def __init__(self, in_channels:int, out_channels:int, p:float=0.5):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout1d(p)
        
        
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x
        

class Model1(nn.Module):
    ''' Directly borrowed from ProxiMic IMUModel8.
    '''
    
    
    def __init__(self) -> None:
        super().__init__()
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
    
    
class Model2(nn.Module):
    
    
    def __init__(self) -> None:
        super().__init__()
        self.mbconv0 = nn.Sequential(MBConv(4, 16, 1, 2, 1), MBConv(16, 24, 6, 1, 2))
        self.mbconv1 = nn.Sequential(MBConv(4, 16, 1, 2, 1), MBConv(16, 24, 6, 1, 2))
        self.mbconv2 = nn.Sequential(MBConv(4, 16, 1, 2, 1), MBConv(16, 24, 6, 1, 2))
        self.mbconv3 = nn.Sequential(MBConv(4, 16, 1, 2, 1), MBConv(16, 24, 6, 1, 2))
        self.mbconv4 = nn.Sequential(MBConv(4, 16, 1, 2, 1), MBConv(16, 24, 6, 1, 2))
        self.mbconv5 = nn.Sequential(MBConv(4, 16, 1, 2, 1), MBConv(16, 24, 6, 1, 2))
        self.conv = nn.Conv1d(in_channels=144, out_channels=24, kernel_size=9, stride=1, padding=4)
        self.pool = nn.MaxPool1d(kernel_size=20)
        self.dense0 = LinearBnDropout(120, 80, p=0.5)
        self.dense1 = LinearBnDropout(80, 40, p=0.5)
        self.dense2 = LinearBnDropout(40, 20, p=0.5)
        self.dense3 = LinearBnDropout(20, 10, p=0.5)
        self.final = nn.Linear(10, cf.N_CLASSES)

    
    def forward(self, x) -> torch.Tensor:
        x0 = self.mbconv0(x[:,0:4,:])
        x1 = self.mbconv0(x[:,4:8,:])
        x2 = self.mbconv0(x[:,8:12,:])
        x3 = self.mbconv0(x[:,12:16,:])
        x4 = self.mbconv0(x[:,16:20,:])
        x5 = self.mbconv0(x[:,20:24,:])
        x = torch.concat([x0,x1,x2,x3,x4,x5], dim=1)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1, 2)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.final(x)
        return x
    
    
class Model3(nn.Module):
    
    
    def __init__(self) -> None:
        super().__init__()
        self.mbconv1 = MBConv(24, 48, 1, 2, 1)
        self.mbconv4 = MBConv(48, 48, 4, 1, 2)
        self.conv = nn.Conv1d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=20)
        self.dense0 = LinearBnDropout(120, 80, p=0.5)
        self.dense1 = LinearBnDropout(80, 40, p=0.5)
        self.dense2 = LinearBnDropout(40, 20, p=0.5)
        self.final = nn.Linear(20, cf.N_CLASSES)
        
        
    def forward(self, x) -> torch.Tensor:
        x = self.mbconv1(x)
        x = self.mbconv4(x)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1, 2)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.final(x)
        return x
    
    
class Model4(nn.Module):
    ''' Change all conv in Model1 to ConvBnAct, all linear to LinearBnDropout,
        with kernel size not changed, to reduce overfit.
    '''
    
    def __init__(self) -> None:
        super().__init__()
        feature_channels = 24
        window_length = int(cf.WINDOW_DURATION * cf.FS_TRAIN)
        self.conv1 = ConvBnAct(in_channels=feature_channels, out_channels=20, kernel_size=1)
        self.conv2 = ConvBnAct(in_channels=window_length, out_channels=20, kernel_size=1)
        self.conv3 = ConvBnAct(in_channels=20, out_channels=20, kernel_size=1)
        self.conv4 = ConvBnAct(in_channels=20, out_channels=20, kernel_size=1)
        self.conv5 = ConvBnAct(in_channels=20, out_channels=20, kernel_size=1)
        self.conv6 = ConvBnAct(in_channels=20, out_channels=20, kernel_size=1)
        self.linear1 = LinearBnDropout(400, 32, p=0.5)
        self.linear2 = nn.Linear(32, cf.N_CLASSES)
        
    
    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x).transpose(1, 2)
        x = self.conv2(x).transpose(1, 2)
        x = (x + self.conv3(x)).transpose(1, 2)
        x = (x + self.conv4(x)).transpose(1, 2)
        x = (x + self.conv5(x)).transpose(1, 2)
        x = x + self.conv6(x)
        x = x.view(-1, 400)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
    
if __name__ == '__main__':
    model = Model1()
    params = model.parameters()
    total = 0
    for item in params:
        print(item.shape, np.prod(item.shape))
        total += np.prod(item.shape)
    print(f'### total: {total // 1000} k')
        