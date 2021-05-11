# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import copy

import random
from mcts_player import Game_State
from mcts_player import get_melds
from GRP.Global_Reward_Prediction import GRP
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.meld import Meld
from mahjong.agari import Agari
from mahjong.hand_calculating.hand_config import OptionalRules

tenhou_option=OptionalRules(has_open_tanyao=True,has_aka_dora=True,has_double_yakuman=False)

class Game_Server(object):
    def __init__(self, setting=None):
        self.TENHOU_HOST = "133.242.10.78"
        self.TENHOU_PORT = 10080
        self.USER_ID = "NoName"
        self.LOBBY = "0"
        self.WAITING_GAME_TIMEOUT_MINUTES = 10
        self.GAME_TYPE = "9"
        if setting:
            #自定义部分
            pass
        
    def self_play(self,players,see_opponent):
        #init_seat_and_score
        grp=GRP('grp_300_extra_3.model')
        score=[250,250,250,250]
        round_number_wind=0
        round_number_dealer=0
        combo=0
        reach_b=0
        #data_set
        action_type=['draw', 'dora', 'discard', 'ron', 'kong', 'pong', 'chow','reach']
        all_data_set={}
        for i in range(8):
            all_data_set[action_type[i]]=[[],[],[]]
        #while not_end
        game_is_continue=True
        while game_is_continue:
            #init_round
            mountion=[i for i in range(136)]
            random.shuffle(mountion)
            hand=[]
            hand.append(mountion[0:13])
            hand.append(mountion[13:26])
            hand.append(mountion[26:39])
            hand.append(mountion[39:52])
            dora=mountion[52]
            mountion=mountion[53:]
            mountion_last=len(mountion)-14
            all_message={}
            all_message['seat']=0
            all_message['opponent']=1
            all_message['player']=0
            all_message['now_id']=round_number_dealer
            all_message['now_opt']=0
            all_message['now_opt_state']=0
            all_message['score']=score[:]
            all_message['combo']=combo
            all_message['reach_b']=reach_b
            all_message['time']=0
            all_message['round']=round_number_wind*4+round_number_dealer
            all_message['wind']=np.array([[[1] if (i//4)==(27+round_number_wind) else [0] for i in range(136)]])
            all_message['dora']=np.array([[[1] if i==dora else [0] for i in range(136)]])
            all_message['closehand']=[]
            all_message['openhand']=[]
            all_message['river']=[]
            all_message['reach']=[]
            all_message['selfwind']=[]
            for i in range(4):
                closehand=np.array([[[1] if (j in hand[i]) else [0] for j in range(136)]])
                openhand=np.array([[[0] for i in range(136)] for j in range(8)])
                river=np.array([[[0] for i in range(136)] for j in range(34)])
                reach=np.array([[[0] for i in range(136)]])
                selfwind=np.array([[[1] if (i//4)==(27+(i-round_number_dealer+4)%4) else [0] for i in range(136)]])
                all_message['closehand'].append(closehand)
                all_message['openhand'].append(openhand)
                all_message['river'].append(river)
                all_message['reach'].append(reach)
                all_message['selfwind'].append(selfwind)
            all_message['mountion']=mountion_last
            all_message['mountion_detail']=np.array([[[1] if (i in mountion) else [0] for i in range(136)]])
            all_message['see_all']=np.array([[[0] if (i in mountion) else [1] for i in range(136)]])
            all_message['last_card']=136
            all_message['river_cnt']=[0,0,0,0]
            all_message['open_cnt']=[0,0,0,0]
            all_state=Game_State(all_message)
            #send_init_round_message
            player_message=[{},{},{},{}]
            for pid in range(4):
                player_message[pid]=copy.deepcopy(all_message)
                player_message[pid]['seat']=pid
                player_message[pid]['now_id']=(round_number_dealer-pid+4)%4
                if pid!=0 :
                    player_message[pid]['score']=player_message[pid]['score'][pid:]+player_message[pid]['score'][0:pid]
                    player_message[pid]['closehand']=player_message[pid]['closehand'][pid:]+player_message[pid]['closehand'][0:pid]
                    player_message[pid]['selfwind']=player_message[pid]['selfwind'][pid:]+player_message[pid]['selfwind'][0:pid]
                if see_opponent==0 :
                    all_message['opponent']=0
                    #closehand
                    #mountion_detail
                    #see_all
                    for i in range(3):
                        for j in range(136):
                            if player_message[pid]['closehand'][i+1][0][j][0]==1:
                                player_message[pid]['closehand'][i+1][0][j][0]==0
                                player_message[pid]['mountion_detail'][0][j][0]=1
                                player_message[pid]['see_all'][0][j][0]=0
                #print(player_message[pid])
                players[pid].set_game_state(player_message[pid])
            #while not_round_end
            data_set={}
            for i in range(8):
                data_set[action_type[i]]=[[],[],[]]
            round_is_continue=True
            card=136
            f = open("train_log.txt", "a")
            print("new round start",file = f)
            for i in range(4):
                print(score[i],[x//4 for x in hand[i]],file = f)
            f.close()
            while round_is_continue:
                _,pid,pop,pos=all_state.get_mct_state()
                f = open("train_log.txt", "a")
                #print(all_state.get_mct_state())
                #for i in range(4):
                    #print(players[i].game_state.get_mct_state())
                print("last action is: ",card,file = f)
                print("round is continue,now is:",pid,pop,pos,file = f)
                for pid_out in range(4):
                    print(score[pid_out],[x//4 for x in range(136) if all_state.closehand[pid_out][0][x][0]==1],[[y//4 for y in range(136) if all_state.openhand[pid_out][x*2][y][0]==1]for x in range(all_state.open_cnt[pid_out])],file = f)
                f.close()
                if pop==0:
                    card=mountion[0]
                    mountion=mountion[1:]
                    for i in range(4):
                        data_set[action_type[i]][0].append(players[i].game_state.get_state_340_136_1())
                        data_set[action_type[i]][1].append(players[i].get_move_probs())
                        data_set[action_type[i]][1].append([1 if j==i else 0 for j in range(4)])
                    for i in range(4):
                        if pid==i or see_opponent==1 :
                            players[i].do_action(card)
                        else:
                            players[i].do_action(136)
                    all_state.do_action(card)
                if pop==1:
                    card=mountion[0]
                    mountion=mountion[1:]
                    for i in range(4):
                        data_set[action_type[i]][0].append(players[i].game_state.get_state_340_136_1())
                        data_set[action_type[i]][1].append(players[i].get_move_probs())
                        data_set[action_type[i]][1].append([1 if j==i else 0 for j in range(4)])
                    for i in range(4):
                        players[i].do_action(card)
                    all_state.do_action(card)
                if pop==2:
                    card=players[pid].get_action()
                    for i in range(4):
                        data_set[action_type[i]][0].append(players[i].game_state.get_state_340_136_1())
                        data_set[action_type[i]][1].append(players[i].get_move_probs())
                        data_set[action_type[i]][1].append([1 if j==i else 0 for j in range(4)])
                    for i in range(4):
                        players[i].do_action(card)
                    all_state.do_action(card)
                if pop==3:
                    for i in range(4):
                        data_set[action_type[i]][0].append(players[i].game_state.get_state_340_136_1())
                        data_set[action_type[i]][1].append(players[i].get_move_probs())
                        data_set[action_type[i]][1].append([1 if j==i else 0 for j in range(4)])
                    if pos==0 or pos==4:
                        card=players[pid].get_action()
                        if card==1:
                            score_d=all_state.get_score_d(pid,pid)
                            cnt=0
                            for sis in score_d:
                                if sis>0:
                                    cnt+=sis
                            if cnt>0:
                                card=1<<pid
                                score_d=score_d[4-pid:]+score_d[0:4-pid]
                                for j in range(4):
                                    score[j]+=score_d[j]
                            else:
                                card=0
                    else :
                        card=0
                        for i in range(3):
                            now_card=players[(pid+1+i)%4].get_action()
                            if now_card==1:
                                score_d=all_state.get_score_d((pid+1+i)%4,pid)
                                cnt=0
                                for sis in score_d:
                                    if sis>0:
                                        cnt+=sis
                                if cnt>0:
                                    card|=(1<<((pid+1+i)%4))
                                    score_d=score_d[4-pid:]+score_d[0:4-pid]
                                    for j in range(4):
                                        score[j]+=score_d[j]
                    for i in range(4):
                        players[i].do_action(card)
                    all_state.do_action(card)
                    if card==0:
                        if pop==2 or pop==6:
                            score[pid]-=10
                            reach_b+=1
                    if card!=0 :
                        reach_b=0
                        round_is_continue=False
                        if card&(1<<round_number_dealer)==1:
                            combo+=1
                        else :
                            round_number_dealer+=1
                            if round_number_dealer==4:
                                round_number_wind+=1
                                round_number_dealer=0
                if pop==4:
                    if pos==0 or pos==2:
                        for i in range(4):
                            data_set[action_type[i]][0].append(players[i].game_state.get_state_340_136_1())
                            data_set[action_type[i]][1].append(players[i].get_move_probs())
                            data_set[action_type[i]][1].append([1 if j==i else 0 for j in range(4)])
                        card=players[pid].get_action()
                        #print(pid,card)
                        for i in range(4):
                            players[i].do_action(card,all_state.opt_is_close_kong(),all_state.last_card//4)
                        all_state.do_action(card,all_state.opt_is_close_kong(),all_state.last_card//4)
                    else:
                        for i in range(4):
                            data_set[action_type[i]][0].append(players[i].game_state.get_state_340_136_1())
                            data_set[action_type[i]][1].append(players[i].get_move_probs())
                            data_set[action_type[i]][1].append([1 if j==i else 0 for j in range(4)])
                        card1=players[(pid+1)%4].get_action()
                        card2=players[(pid+1)%4].get_action()
                        card3=players[(pid+1)%4].get_action()
                        card=[0,0,0]
                        for i in range(3):
                            if card1[i]!=0:
                                card[i]=card1[i]
                                break
                            if card2[i]!=0:
                                card[i]=card2[i]
                                break
                            if card3[i]!=0:
                                card[i]=card3[i]
                                break
                        for j in range(3):
                            for i in range(4):
                                data_set[action_type[i]][0].append(players[i].game_state.get_state_340_136_1())
                                data_set[action_type[i]][1].append(players[i].get_move_probs())
                                data_set[action_type[i]][1].append([1 if j==i else 0 for j in range(4)])
                            for i in range(4):
                                players[i].do_action(card[j])
                            all_state.do_action(card[j])
                            if card[j]!=0:
                                break
                if pop==7:
                    card=players[pid].get_action()
                    for i in range(4):
                        data_set[action_type[i]][0].append(players[i].game_state.get_state_340_136_1())
                        data_set[action_type[i]][1].append(players[i].get_move_probs())
                        data_set[action_type[i]][1].append(np.array([1 if j==i else 0 for j in range(4)]))
                    for i in range(4):
                        players[i].do_action(card)
                    all_state.do_action(card)
                if not all_state.not_mountion_empty():
                    round_is_continue=False
                    combo+=1
                    round_number_dealer+=1
                    if round_number_dealer==4:
                        round_number_wind+=1
                        round_number_dealer=0
                    break
            #_get_one_part_data
            score_p=grp.get_reward(np.array([[round_number_wind,round_number_dealer,combo,score[0],score[1],score[3],score[2]]]))[0]
            for i in range(8):
                l=len(data_set[action_type[i]][2])
                for j in range(l):
                    pid=[i for i in range(4) if data_set[action_type[i]][2][i]==1][0]
                    score_pid=score_p[pid:]+score_p[0:pid]
                    for k in range(4):
                        data_set[action_type[i]][2][k]=score_pid[k]
            for i in range(8):
                all_data_set[action_type[i]][0]=all_data_set[action_type[i]][0]+data_set[action_type[i]][0]
                all_data_set[action_type[i]][1]=all_data_set[action_type[i]][0]+data_set[action_type[i]][1]
                all_data_set[action_type[i]][2]=all_data_set[action_type[i]][0]+data_set[action_type[i]][2]
            if round_number_wind==2:
                game_is_continue=False
        #send_data
        return data_set
    def online_play(self,player):
        #get conect
        #search game
        #start game
        #end game
        pass