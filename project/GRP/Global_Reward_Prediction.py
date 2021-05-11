# -*- coding: utf-8 -*-

from __future__ import print_function
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

class GRP_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.val_fc1 = nn.Linear(7, 32)
        self.val_fc2 = nn.Linear(32, 128)
        
        self.extra_fc1 = nn.Linear(128,512)
        self.extra_fc2 = nn.Linear(512,512)
        self.extra_fc3 = nn.Linear(512,512)
        self.extra_fc4 = nn.Linear(512,512)
        self.extra_fc5 = nn.Linear(512,512)
        self.extra_fc6 = nn.Linear(512,128)
        
        self.val_fc3 = nn.Linear(128, 32)
        self.val_fc4 = nn.Linear(32, 4)
        
    def forward(self, state_input):
        x = F.relu(self.val_fc1(state_input))
        x = F.relu(self.val_fc2(x))
        
        x = F.relu(self.extra_fc1(x))
        x = F.relu(self.extra_fc2(x))
        x = F.relu(self.extra_fc3(x))
        x = F.relu(self.extra_fc4(x))
        x = F.relu(self.extra_fc5(x))
        x = F.relu(self.extra_fc6(x))
        
        x = F.relu(self.val_fc3(x))
        x = F.tanh(self.val_fc4(x))
        return x
        

class GRP():
    def __init__(self, model_file=None, use_gpu=True):
        self.use_gpu=use_gpu
        if self.use_gpu:
            self.grp_net=GRP_Net().cuda()
        else :
            self.grp_net=GRP_Net()
        self.l2_const = 1e-4
        self.optimizer = optim.Adam(self.grp_net.parameters(),weight_decay=self.l2_const)
        if model_file:
            net_params = torch.load(model_file)
            self.grp_net.load_state_dict(net_params)
        
    def get_reward(self, state_input):
        if self.use_gpu:
            state_input = Variable(torch.FloatTensor(state_input).cuda())
        else :
            state_input = Variable(torch.FloatTensor(state_input))
        predicts = self.grp_net(state_input)
        return predicts.detach().cpu().numpy()
    
    def get_loss(self,state_batch, target_batch):
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            target_batch = Variable(torch.FloatTensor(target_batch).cuda())
        else :
            state_batch = Variable(torch.FloatTensor(state_batch))
            target_batch = Variable(torch.FloatTensor(target_batch))
        # forward
        predicts = self.grp_net(state_batch)
        loss = F.mse_loss(predicts, target_batch)
        return loss.item()
    
    def train_step(self, state_batch, target_batch):
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            target_batch = Variable(torch.FloatTensor(target_batch).cuda())
        else :
            state_batch = Variable(torch.FloatTensor(state_batch))
            target_batch = Variable(torch.FloatTensor(target_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward
        predicts = self.grp_net(state_batch)
        # define the loss = (z - v)^2 + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        loss = F.mse_loss(predicts, target_batch)
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_param(self):
        net_params = self.grp_net.state_dict()
        return net_params
    
    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_param()  # get model params
        torch.save(net_params, model_file)
    
if __name__ == '__main__':
    train_dat = np.load('train_dat.npy', allow_pickle=True);
    train_len = len(train_dat)
    test_len = train_len // 10
    test_dat = train_dat[0:test_len]
    train_dat = train_dat[test_len:]
    train_len -= test_len
    
    # '''
    # train
    grp = GRP()
    for i in range(300):
        choiced_idx = np.random.choice(train_len,512,replace=False)
        # print(choiced_idx)
        state_batch = []
        target_batch = []
        for idx in choiced_idx:
            state_batch.append(train_dat[idx][0])
            target_batch.append(train_dat[idx][1])
        state_batch = np.array(state_batch)
        target_batch = np.array(target_batch)
        # print(grp.get_loss(state_batch, target_batch))
        print(i,grp.train_step(state_batch, target_batch))
    grp.save_model('grp_300_extra_4.model')
    # '''
    # '''
    # test
    # grp = GRP('grp_1.model')
    for i in range(4):
        choiced_idx = np.random.choice(test_len,512,replace=False)
        # print(choiced_idx)
        state_batch = []
        target_batch = []
        for idx in choiced_idx:
            state_batch.append(train_dat[idx][0])
            target_batch.append(train_dat[idx][1])
        state_batch = np.array(state_batch)
        target_batch = np.array(target_batch)
        print(grp.get_loss(state_batch, target_batch))
        choiced_idx = np.random.choice(512,16,replace=False)
        state = []
        target = []
        for idx in choiced_idx:
            state.append(state_batch[idx])
            target.append(target_batch[idx])
        state = np.array(state)
        target = np.array(target)
        predict = grp.get_reward(state)
        for j in range(16):
            print(state[j],target[j],predict[j])
    # '''
