# -*- coding: utf-8 -*-

# 32*34=1088

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels ,kernel_size=(3,1), padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels ,kernel_size=(3,1), padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out += residual
        return out
        

class Net(nn.Module):
    def __init__(self, num_output = None):
        super(Net, self).__init__()
        
        self.num_output = num_output
        # common layers
        self.conv1 = nn.Conv2d(920,256,kernel_size=(3,1), padding=1)
        self.res1 = ResidualBlock(256,256)
        self.res2 = ResidualBlock(256,256)
        self.res3 = ResidualBlock(256,256)
        self.res4 = ResidualBlock(256,256)
        self.res5 = ResidualBlock(256,256)
        # action policy layers
        if num_output:
            self.act_conv1 = nn.Conv2d(256, 32,kernel_size=(3,1), padding=1)
            self.act_fc1 = nn.Linear(1088, 1024)
            self.act_fc2 = nn.Linear(1024, 256)
            self.act_fc3 = nn.Linear(256, num_output)
        else :
            self.act_conv1 = nn.Conv2d(256, 1, kernel_size=1)
            self.act_fc1 = nn.Linear(34, 34)
        # state value layers
        self.val_conv1 = nn.Conv2d(256, 32,kernel_size=(3,1), padding=1)
        self.val_fc1 = nn.Linear(1088, 1024)
        self.val_fc2 = nn.Linear(1024, 256)
        self.val_fc3 = nn.Linear(256, 4) 
        
    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.res3(x))
        x = F.relu(self.res4(x))
        x = F.relu(self.res5(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        if self.num_output:
            x_act = x_act.view(-1, 1088)
            x_act = F.relu(self.act_fc1(x_act))
            x_act = F.relu(self.act_fc2(x_act))
            x_act = F.relu(self.act_fc3(x_act))
        else :
            x_act = x_act.view(-1, 34)
            x_act = F.relu(self.act_fc1(x_act))
        x_act = F.log_softmax(x_act)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 1088)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.relu(self.val_fc2(x_val))
        x_val = F.tanh(self.val_fc3(x_val))
        return x_act, x_val

