"""
Base on tutorial :https://towardsdev.com/original-u-net-in-pytorch-ebe7bb705cc7
"""

import torch
import torch.nn as nn

class DownConvolution(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(DownConvolution,self).__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel,(3,3),padding='same')
        self.conv2 = nn.Conv2d(output_channel,output_channel,(3,3),padding='same')
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = self.maxpooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x