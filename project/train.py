# -*- coding: utf-8 -*-

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from mcts_player import MCTSPlayer
from Player_Net.ai_net import PolicyValueNet 
from game import Game_Server


action_type=['draw', 'dora', 'discard', 'ron', 'kong', 'pong', 'chow','reach']

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = [deque(maxlen=self.buffer_size) for i in range(8)]
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        if init_model:
            self.policy_value_net = PolicyValueNet(model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet()
        self.players = [MCTSPlayer(self.policy_value_net.policy_value_fn,c_puct=self.c_puct,n_playout=self.n_playout,is_selfplay=1) for i in range(4)]
        self.game=Game_Server()


    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            play_data = self.game.self_play(self.players,see_opponent=1)
            self.episode_len=0
            for i in range(8):
                dat[0]=play_data[action_type[i]][0][:]
                dat[1]=play_data[action_type[i]][1][:]
                dat[1]=play_data[action_type[i]][2][:]
                self.episode_len+=len(dat[0])
                self.data_buffer[i].extend(zip(dat[0],dat[1],dat[2]))

    def policy_update(self,idx):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer[idx], self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        score_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch,action_type[idx])
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    score_batch,
                    action_type[idx],
                    self.learn_rate*self.lr_multiplier[idx])
            new_probs, new_v = self.policy_value_net.policy_value(state_batch,action_type[idx])
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier[idx] > 0.1:
            self.lr_multiplier[idx] /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier[idx] < 10:
            self.lr_multiplier[idx] *= 1.5
        print(("action_type:{}"
               "kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               ).format(action_type[idx],
                        kl,
                        self.lr_multiplier[idx],
                        loss,
                        entropy))
        return action_type[idx], loss, entropy

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                f = open("train_log.txt", "a")
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len),file = f)
                f.close()
                for i in range(8):
                    if len(self.data_buffer[i]) > self.batch_size:
                        act_tp, loss, entropy = self.policy_update(i)
                # check the performance of the current model, and save the model params
                if (i+1) % self.check_freq == 0:
                    f = open("train_log.txt", "a")
                    print("current self-play batch: {}".format(i+1),file = f)
                    f.close()
                    self.policy_value_net.save_model('./current_policy.model')
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
