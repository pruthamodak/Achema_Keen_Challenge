from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init
import os

class ConvBlock(nn.Module):
    def __init__(self,
                in_channels: int, 
                out_channels: int,
                kernel_size: int=3, 
                stride: int=1,
                padding=0,
                bias: bool = True) -> None:
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, inp):
        identity = self.conv1(inp)
        identity = self.bn1(identity)
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        return self.relu(x)

class KeenModel(nn.Module):
    def __init__(self, num_classes, input_shape, factor=1):
        super(KeenModel, self).__init__()

        input_channels = 3
        # initial convolution
        self.conv1 = ConvBlock(input_channels, 64//factor, stride=2, padding=1)
        self.conv2 = ConvBlock(64//factor, 128 // factor, stride=2, padding=1)
        self.conv3 = ConvBlock(128 // factor, 256 // factor, stride=2, padding=1)
        self.conv4 = ConvBlock(256 // factor, 512 // factor, stride=2, padding=1)
        self.conv5 = ConvBlock(512 // factor, 512 // factor, stride=2, padding=1)         
        self.fc1 = nn.Linear((512//factor) * (input_shape//32) * (input_shape//32), 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.kaiming_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

