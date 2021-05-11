# -*- coding: utf-8 -*-

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
        self.conv1 = nn.Conv2d(in_channels, out_channels ,kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels ,kernel_size=(3,3), padding=1)

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
        self.in_channel_num=196 #340
        self.tile_type=136
        self.tile_type_32=self.tile_type*32
        self.num_output = num_output
        # common layers
        self.conv1 = nn.Conv2d(self.in_channel_num,256,kernel_size=(3,3), padding=1)
        self.res1 = ResidualBlock(256,256)
        self.res2 = ResidualBlock(256,256)
        self.res3 = ResidualBlock(256,256)
        self.res4 = ResidualBlock(256,256)
        self.res5 = ResidualBlock(256,256)
        # action policy layers
        if num_output:
            self.act_conv1 = nn.Conv2d(256, 32,kernel_size=(3,3), padding=1)
            self.act_fc1 = nn.Linear(self.tile_type_32, 1024)
            self.act_fc2 = nn.Linear(1024, 256)
            self.act_fc3 = nn.Linear(256, num_output)
        else :
            self.act_conv1 = nn.Conv2d(256, 1, kernel_size=1)
            self.act_fc1 = nn.Linear(self.tile_type, self.tile_type)
        # state value layers
        self.val_conv1 = nn.Conv2d(256, 32,kernel_size=(3,3), padding=1)
        self.val_fc1 = nn.Linear(self.tile_type_32, 1024)
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
            x_act = x_act.view(-1, self.tile_type_32)
            x_act = F.relu(self.act_fc1(x_act))
            x_act = F.relu(self.act_fc2(x_act))
            x_act = F.relu(self.act_fc3(x_act))
        else :
            x_act = x_act.view(-1, self.tile_type)
            x_act = F.relu(self.act_fc1(x_act))
        x_act = F.log_softmax(x_act)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.tile_type_32)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.relu(self.val_fc2(x_val))
        x_val = F.tanh(self.val_fc3(x_val))
        return x_act, x_val



class PolicyValueNet():
    """policy-value network """
    def __init__(self, model_file=None, use_gpu=True):
        self.in_channel_num=196 # 340
        self.tile_type=136
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        # draw/ron/discard/kong/pong/chow/dora
        if self.use_gpu:
            self.draw_net = Net().cuda()
            self.dora_net = Net().cuda()
            self.discard_net = Net().cuda()
            self.ron_net = Net(num_output=16).cuda()
            self.kong_net = Net(num_output=5).cuda()
            self.pong_net = Net(num_output=13).cuda()
            self.chow_net = Net(num_output=49).cuda()
            self.reach_net = Net(num_output=2).cuda()
        else:
            self.draw_net = Net()
            self.dora_net = Net()
            self.discard_net = Net()
            self.ron_net = Net(num_output=16)
            self.kong_net = Net(num_output=5)
            self.pong_net = Net(num_output=13)
            self.chow_net = Net(num_output=49)
            self.reach_net = Net(num_output=2)
            
        self.optimizer_draw = optim.Adam(self.draw_net.parameters(), weight_decay=self.l2_const)
        self.optimizer_dora = optim.Adam(self.dora_net.parameters(), weight_decay=self.l2_const)
        self.optimizer_discard = optim.Adam(self.discard_net.parameters(), weight_decay=self.l2_const)
        self.optimizer_ron = optim.Adam(self.ron_net.parameters(), weight_decay=self.l2_const)
        self.optimizer_kong = optim.Adam(self.kong_net.parameters(), weight_decay=self.l2_const)
        self.optimizer_pong = optim.Adam(self.pong_net.parameters(), weight_decay=self.l2_const)
        self.optimizer_chow = optim.Adam(self.chow_net.parameters(), weight_decay=self.l2_const)
        self.optimizer_reach = optim.Adam(self.reach_net.parameters(), weight_decay=self.l2_const)
        self.nets = {'draw' : {'net' : self.draw_net, 'optim' : self.optimizer_draw},
                     'dora' : {'net' : self.dora_net, 'optim' : self.optimizer_dora},
                     'discard' : {'net' : self.discard_net, 'optim' : self.optimizer_discard}, 
                    'ron' : {'net' : self.ron_net, 'optim' : self.optimizer_ron},
                     'kong' : {'net' : self.kong_net, 'optim' : self.optimizer_kong},
                     'pong' : {'net' : self.pong_net, 'optim' : self.optimizer_pong}, 
                    'chow' : {'net' : self.chow_net, 'optim' : self.optimizer_chow},
                    'reach' : {'net' : self.reach_net, 'optim' : self.optimizer_reach},}
        self.action_types=['draw', 'dora', 'discard', 'ron', 'kong', 'pong', 'chow','reach']
        if model_file:
            self.load_model()

    def policy_value(self, state_batch, action_type):
        """data 改动 detach()/item()"""
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            # log_act_probs, value = self.policy_value_net(state_batch)
            log_act_probs, value = self.nets[action_type]['net'](state_batch)
            act_probs = np.exp(log_act_probs.detach().cpu().numpy())
            return act_probs, value.detach().cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            # log_act_probs, value = self.policy_value_net(state_batch)
            log_act_probs, value = self.nets[action_type]['net'](state_batch)
            act_probs = np.exp(log_act_probs.detach().numpy())
            return act_probs, value.detach().numpy()

    def policy_value_fn(self, game_state):
        """
        data 改动 detach()/item()
        input: game_state
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        current_state, action_type, legal_positions = game_state.current_state()
        current_state = np.ascontiguousarray(current_state.reshape(-1, self.in_channel_num, self.tile_type, 1))
        if self.use_gpu:
            log_act_probs, value = self.nets[action_type]['net'](Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())
            value = value.detach().cpu().numpy().flatten()
        else:
            log_act_probs, value = self.nets[action_type]['net'](Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.detach().numpy().flatten())
            value = value.detach().numpy().flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, score_batch, action_type, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            score_batch = Variable(torch.FloatTensor(score_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            score_batch = Variable(torch.FloatTensor(score_batch))
        
        # zero the parameter gradients
        self.nets[action_type]['optim'].zero_grad()
        # set learning rate
        set_learning_rate(self.nets[action_type]['optim'], lr)

        # forward
        log_act_probs, value = self.nets[action_type]['net'](state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value, score_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.nets[action_type]['optim'].step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def save_model(self, model_file):
        """ save model params to file """
        net_params = {}
        for action_type in self.action_types:
            net_params[action_type]=self.nets[action_type]['net'].state_dict()
        torch.save(net_params, model_file)
        
    def load_model(self, model_file):
        """ load model params from file """
        net_params = torch.load(model_file)
        for action_type in self.action_types:
            self.nets[action_type]['net'].load_state_dict(net_params[action_type])
