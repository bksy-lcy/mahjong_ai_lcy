# -*- coding: utf-8 -*-

import numpy as np
import copy
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.meld import Meld
from mahjong.agari import Agari
from mahjong.hand_calculating.hand_config import OptionalRules

tenhou_option=OptionalRules(has_open_tanyao=True,has_aka_dora=True,has_double_yakuman=False)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def tile_type(x):
    if x<9:
        return 0
    if x<18:
        return 1
    if x<27:
        return 2
    return 3

def get_melds(openhand,open_cnt):
    if open_cnt==0:
        return []
    melds=[]
    for i in range(open_cnt):
        tile=[j for j in range(136) if openhand[i*2][j][0]==1]
        opens=[j for j in range(136) if openhand[i*2+1][j][0]==1]
        melds.append(Meld(tiles=tile, opened=(opens!=[])))
    return melds

# opt: 0,draw 1,dora 2,dsicard 3,ron 4,kong 5,pong 6,chow 7,reach 8,end
def MCT_Nex(player,opt,state,choice,close_kong=0):
    """
    状态转移函数
    """
    if opt==0 :
        if state==0 :
            return player,3,0
        if state==1 :
            return player,3,4
    if opt==1 :
        if state==0 :
            return player,0,0
        if state==1 :
            return (player+1)%4,0,0
        if state==2 :
            return player,1,0
        if state==3 :
            return player,3,7
        if state==4 :
            return player,0,1
        if state==5:
            return player,2,0
    if opt==2 :
        if state==0 :
            return player,3,1
        if state==1 :
            return player,3,2
        if state==2 :
            return player,3,5
        if state==3 :
            return player,3,6
    if opt==3 :
        if state==0 :
            if choice==0 :
                return player,4,0
            else :
                return player,8,0
        if state==1 :
            if choice==0 :
                return player,4,1
            else :
                return player,8,0
        if state==2 :
            if choice==0 :
                return player,4,1
            else :
                return player,8,0
        if state==3 :
            if choice==0 :
                return player,0,1
            else :
                return player,8,0
        if state==4 :
            if choice==0 :
                return player,4,2
            else :
                return player,8,0
        if state==5 :
            if choice==0 :
                return player,4,3
            else :
                return player,8,0
        if state==6 :
            if choice==0 :
                return player,4,3
            else :
                return player,8,0
        if state==7 :
            if choice==0 :
                return player,0,1
            else :
                return player,8,0
    if opt==4 :
        if state==0:
            if choice==0 :
                return player,7,0
            else :
                if close_kong==1:
                    return player,1,0
                else :
                    return player,3,3
        if state==1 :
            if choice==0 :
                return player,5,0
            else :
                return (player+choice-1)%4,0,1
        if state==2:
            if choice==0 :
                return player,7,1
            else :
                if close_kong==1:
                    return player,1,2
                else :
                    return player,1,3
        if state==3:
            if choice==0 :
                return player,5,1
            else :
                return (player+choice-1)%4,1,4
    if opt==5 :
        if state==0:
            if choice==0 :
                return player,6,0
            else :
                return (player+(choice-1)//4)%4,2,0
        if state==1:
            if choice==0 :
                return player,6,1
            else :
                return (player+(choice-1)//4)%4,1,5
    if opt==6 :
        if state==0:
            if choice==0 :
                return (player+1)%4,0,0
            else :
                return (player+1)%4,2,0
        if state==1:
            if choice==0 :
                return player,1,1
            else :
                return (player+1)%4,1,5
    if opt==7 :
        if state==0 :
            if choice==0 :
                return player,2,0
            else :
                return player,2,1
        if state==1 :
            if choice==0 :
                return player,2,2
            else :
                return player,2,3
    return -1,-1,-1

class MCT_TreeNode(object):
    def __init__(self, _parent, _prior_p, _main_player_id, _now_player_id, _now_player_opt, _now_player_opt_state):
        """
        _parent:父节点
        _prior_p:先验概率
        _main_player_id:主视角id(0-3)
        _now_player_id:当前玩家id(0-3)
        _now_player_opt:当前玩家正在进行的操作(0-6)
        _now_player_opt_state:当前玩家正在进行的操作的种类(0-3)
        """
        self._parent=_parent
        self._P=_prior_p
        self._main_player_id=_main_player_id # 恒为0 # 懒得改了
        self._now_player_id=_now_player_id
        self._now_player_opt=_now_player_opt
        self._now_player_opt_state=_now_player_opt_state
        self._children={}
        self._n_visits=0
        self._Q=[0,0,0,0]
        self._u=0
        
    def expand(self,action_priors,close_kong=0):
        for action, prob in action_priors:
            if action not in self._children:
                mxn=MCT_Nex(self._now_player_id,self._now_player_opt,self._now_player_opt_state,action,close_kong)
                self._children[action] = MCT_TreeNode(self,prob,self._main_player_id,mxn[0],mxn[1],mxn[2])
    
    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        for i in range(4):
            self._Q[i] += 1.0*(leaf_value[i] - self._Q[i]) / self._n_visits
    
    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)
    
    def get_value(self, c_puct):
        #
        pid=self._parent._now_player_id
        pop=self._parent._now_player_opt
        pos=self._parent._now_player_opt_state
        player_id=0
        if pop==2 or (pop==4 and pos!=1) or (pop==3 and (pos==0 or pos==4)) or pop==7:
            player_id=pid
        else :
            if pop==0 or pop==1 or pid==0 or (pop==6 and pid!=3):
                c_puct=100
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q[player_id] + self._u
    
    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct, n_playout, init_state):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = MCT_TreeNode(None, 1.0,init_state[0],init_state[1],init_state[2],init_state[3])
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_action(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        if node._now_player_opt!=8 and state.not_mountion_empty():
            # not end
            node.expand(action_probs,state.opt_is_close_kong())
        
        if node._now_player_opt!=8 and not state.not_mountion_empty():
            leaf_value=[0,0,0,0]
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs
    def get_move_probs_pong(self, temp=1e-3):
        if 0 in self._root._children:
            now_root=self._root._children[0]
            if not now_root.is_leaf():
                act_visits = [(act, node._n_visits) for act, node in now_root._children.items()]
                acts, visits = zip(*act_visits)
                act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
                return acts, act_probs
        return [0],[0]
    
    def get_move_probs_chow(self, temp=1e-3):
        if 0 in self._root._children:
            now_root=self._root._children[0]
            if not now_root.is_leaf():
                if 0 in now_root._children:
                    now_root=now_root._children[0]
                    if not now_root.is_leaf():
                        act_visits = [(act, node._n_visits) for act, node in now_root._children.items()]
                        acts, visits = zip(*act_visits)
                        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
                        return acts, act_probs
        return [0],[0]
        
    def update_with_move(self, last_move, close_kong):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            mxn=MCT_Nex(self._root._now_player_id,self._root._now_player_opt,self._root._now_player_opt_state,last_move,close_kong)
            self._root = MCT_TreeNode(None, 1.0, self._root._main_player_id,mxn[0],mxn[1],mxn[2])

    def __str__(self):
        return "MCTS"

class Game_State(object):
    def __init__(self, init_message):
        self.opt_num_move=[136,136,136,16,5,13,49,2,0]
        self.opt_name=['draw','dora','discard','ron','kong','pong','chow','reach','end']
        self.seat=init_message['seat']# 0-3
        self.see_opponent=init_message['opponent']# 0/1
        
        self.main_player_id=init_message['player']# 0
        self.now_player_id=init_message['now_id']# 0-3
        self.now_player_opt=init_message['now_opt']# 0-8
        self.now_player_opt_state=init_message['now_opt_state']# 0-7
        
        self.score=init_message['score']# 4*int 4*136*1
        self.combo=init_message['combo']# int 1*136*1
        self.reach_b=init_message['reach_b']# int 1*136*1
        self.time=init_message['time']# int 1*136*1
        #seat 1*136*1
        self.round=init_message['round']# int 1*136*1
        #now_id 1*136*1
        self.wind=init_message['wind']# 1*136*1
        self.dora=init_message['dora']# 1*136*1
        
        self.closehand=init_message['closehand']#4*1*136*1
        self.openhand=init_message['openhand']#4*8(4*2)*136*1
        self.river=init_message['river']#4*34*136*1
        self.reach=init_message['reach']#4*1*136*1
        self.selfwind=init_message['selfwind']#4*1*136*1
        ## 得点与速度 4*36(3*12)*136*1
        
        self.mountion=init_message['mountion']#int 1*136*1
        self.mountion_detail=init_message['mountion_detail']#1*136*1
        self.see_all=init_message['see_all']#1*136*1
        self.last_card=init_message['last_card']#int 1*36*1
        #共340*136*1
        
        #不在网络输入里，辅助局面转移
        self.point_target=[1000,2000,3000,4000,6000,8000,12000,16000,18000,24000,32000,48000]
        self.river_cnt=init_message['river_cnt']#4*int
        self.open_cnt=init_message['open_cnt']
        
    def do_action(self,action,close_kong=0,extra=None):
        nex_mct_state=MCT_Nex(self.now_player_id,self.now_player_opt,self.now_player_opt_state,action,close_kong)
        #change state
        if self.now_player_opt==0 :
            self.last_card=action
            self.mountion-=1
            if self.river_cnt[self.now_player_id]==self.time :
                self.time+=1
            if self.see_opponent==1 or self.now_player_id==0 :
                self.closehand[self.now_player_id][0][action][0]=1
                self.see_all[0][action][0]=1
                self.mountion_detail[0][action][0]=0
            
        if self.now_player_opt==1 :
            self.dora[0][action][0]=1
            self.see_all[0][action][0]=1
            self.mountion_detail[0][action][0]=0
            
        if self.now_player_opt==2 :
            self.last_card=action
            if self.see_opponent==1 or self.now_player_id==0 :
                self.closehand[self.now_player_id][0][action][0]=0
            self.river[self.now_player_id][self.time][action][0]=1
            self.river_cnt[self.now_player_id]+=1
            self.see_all[0][action][0]=1
            self.mountion_detail[0][action][0]=0
            

        if self.now_player_opt==3:
            if action==0 :
                if self.now_player_opt_state==2 or self.now_player_opt_state==6 :
                    self.reach_b+=1
                    self.score[self.now_player_id]-=10 # 10*100
                else :
                    #无改变
                    pass
            else :
                #游戏结束
                pass
                        
        if self.now_player_opt==4:
            if action==0 :
                #无改变
                pass
            else :
                next_player=(self.now_player_id+(action-1))%4
                #视extra而定
                if extra is None :
                    card_type=self.last_card//4
                    for i in range(4):
                        self.openhand[next_player][self.open_cnt[next_player]*2][card_type*4+i][0]=1
                        self.closehand[next_player][0][card_type*4+i][0]=0
                    self.openhand[next_player][self.open_cnt[next_player]*2+1][self.last_card][0]=1
                    self.open_cnt[next_player]+=1
                else :
                    card_type=extra
                    if close_kong :
                        for i in range(4):
                            self.openhand[next_player][self.open_cnt[next_player]*2][card_type*4+i][0]=1
                            self.closehand[next_player][0][card_type*4+i][0]=0
                        self.open_cnt[next_player]+=1
                    else :
                        for i in range(self.open_cnt[next_player]):
                            if self.openhand[next_player][i*2][card_type*4][0]==1 or self.openhand[next_player][i*2][card_type*4+1][0]==1 :
                                for j in range(4):
                                    if self.openhand[next_player][i*2][card_type*4+j][0]==0:
                                        self.last_card=card_type*4+j
                                    self.openhand[next_player][i*2][card_type*4+j][0]=1
                                break
                for i in range(4):
                    self.see_all[0][card_type*4+i][0]=1
            
        if self.now_player_opt==5:
            if action==0 :
                #无改变
                pass
            else :
                next_player=(self.now_player_id+(action-1)//4)%4
                action_type=(action-1)%4
                card_type=self.last_card//4
                for i in range(4):
                    if i==action_type:
                        continue
                    self.openhand[next_player][self.open_cnt[next_player]*2][card_type*4+i][0]=1
                    self.closehand[next_player][0][card_type*4+i][0]=0
                    self.see_all[0][card_type*4+i][0]=1
                self.openhand[next_player][self.open_cnt[next_player]*2+1][self.last_card][0]=1
                self.open_cnt[next_player]+=1
            
        if self.now_player_opt==6:
            if action==0 :
                #无改变
                pass
            else :
                next_player=(self.now_player_id+1)%4
                action_type=action-1
                chow_type=action_type//16
                d1=(action_type%16)//4
                d2=action_type%4
                card_type=self.last_card//4
                card1_d=[1,-1,-2]
                card2_d=[2,1,-1]
                card1=card_type+card1_d[chow_type]
                card2=card_type+card2_d[chow_type]
                card1=card1*4+d1
                card2=card2*4+d2
                self.closehand[next_player][0][card1][0]=0
                self.closehand[next_player][0][card2][0]=0
                self.openhand[next_player][self.open_cnt[next_player]*2][card1][0]=1
                self.openhand[next_player][self.open_cnt[next_player]*2][card2][0]=1
                self.openhand[next_player][self.open_cnt[next_player]*2][self.last_card][0]=1
                self.see_all[0][card1][0]=1
                self.see_all[0][card2][0]=1
                self.see_all[0][self.last_card][0]=1
                self.openhand[next_player][self.open_cnt[next_player]*2+1][self.last_card][0]=1
                self.open_cnt[next_player]+=1
                            
        if self.now_player_opt==7:
            if action==0 :
                #无改变
                pass
            else :
                self.reach[self.now_player_id][0][self.time][0]=1    
        self.now_player_id=nex_mct_state[0]
        self.now_player_opt=nex_mct_state[1]
        self.now_player_opt_state=nex_mct_state[2]
        
    def get_num_move(self):
        return self.opt_num_move[self.now_player_opt]
    
    def not_mountion_empty(self):
        return self.mountion!=0
        
    def opt_is_close_kong(self):
        if self.now_player_opt!=2 :
            return 0
        if self.now_player_opt_state==1 :
            return 0
        card_type=(self.last_card//4)*4
        if self.see_opponent or self.now_player_id==0 :
            for i in range(4):
                if self.closehand[self.now_player_id][0][card_type+i]==0 :
                    return 0
        else :
            for i in range(4):
                if self.see_all[card_type+i]==1 :
                    return 0
        return 1
    
    def get_mct_state(self):
        return self.seat,self.now_player_id,self.now_player_opt,self.now_player_opt_state
        
        
    def check_get_point_subfunc(self,now_closehand,new_card,now_openhand,now_open_cnt,now_reach,now_reach_b,now_combo,now_wind,now_selfwind,now_dora):
        #import mahjong
        closehand=[i for i in range(136) if now_closehand[i]==1]
        melds=get_melds(now_openhand,now_open_cnt)
        for m in melds:
            for tile in m.tiles:
                closehand.append(tile)
        dora=[i for i in range(136) if now_dora[i][0]==1]
        wind=[i for i in range(136) if now_wind[i][0]==1][0]//4
        selfwind=[i for i in range(136) if now_selfwind[i][0]==1][0]//4
        calculator = HandCalculator()
        result = calculator.estimate_hand_value(closehand,new_card,melds,dora,HandConfig(is_tsumo=True,is_riichi=(now_reach==1) ,kyoutaku_number=now_reach_b,tsumi_number=now_combo,player_wind=selfwind,round_wind=wind,options=tenhou_option))
        if result.cost:
            return result.cost['total']
        else :
            return 0
        
    def check_get_point(self,player,time,discard,target):
        now_closehand=[1 if self.closehand[player][0][i][0]==1 else 0 for i in range(136)]
        now_close_cnt=0
        for i in range(136):
            if now_closehand[i]:
                now_close_cnt+=1
        now_openhand=self.openhand[player]
        now_open_cnt=self.open_cnt[player]
        now_reach=0
        for i in range(136):
            if self.reach[player][0][i][0]==1:
                now_reach=1
        now_reach_b=self.reach_b
        now_combo=self.combo
        now_wind=self.wind[0]
        now_selfwind=self.selfwind[player][0]
        now_dora=self.dora[0]
        now_mountion=[1 if self.mountion_detail[0][i][0]==1 else 0 for i in range(136)]
        #import mahjong
        if now_close_cnt+now_open_cnt*3==14 :
            if now_closehand[discard]==0 :
                return 0
            now_closehand[discard]=0
        for i in range(136):
            if now_mountion[i]==0:
                continue
            now_closehand[i]=1
            if self.check_get_point_subfunc(now_closehand,i,now_openhand,now_open_cnt,now_reach,now_reach_b,now_combo,now_wind,now_selfwind,now_dora)>=target :
                return 1
            if time>0 :
                now_mountion[i]=0
                for j in range(136):
                    if now_closehand[j]==0:
                        continue
                    now_closehand[j]=0
                    for k in range(136):
                        if now_mountion[k]==0:
                            continue
                            now_closehand[k]=1
                            if self.check_get_point_subfunc(now_closehand,k,now_openhand,now_open_cnt,now_reach_b,now_combo,now_wind,now_selfwind,now_dora)>=target :
                                return 1
                            if time>1 :
                                now_mountion[k]=0
                                for l in range(136):
                                    if now_closehand[l]==0:
                                        continue
                                    now_closehand[l]=0
                                    for m in range(136):
                                        if now_mountion[m]==0:
                                            continue
                                        now_closehand[m]=1
                                        if self.check_get_point_subfunc(now_closehand,m,now_openhand,now_open_cnt,now_reach_b,now_combo,now_wind,now_selfwind,now_dora)>=target :
                                            return 1
                                        now_closehand[m]=0
                                    now_closehand[l]=1
                                now_mountion[k]=1
                            now_closehand[k]=0
                    now_closehand[j]=1
                now_mountion[i]=1
            now_closehand[i]=0
        return 0
    
    def get_gain_speed(self,player):
        gs=[]
        if player!=0 and self.see_opponent==0 :
            for i in range(36):
                gs.append([[0] for j in range(136)])
        else :
            for i in range(2):
                for j in range(12):
                    gs.append([[1] if self.check_get_point(player,i,k,self.point_target[j]) else [0] for k in range(136)])
        return gs
        
    def get_state_340_136_1(self):
        now_state=[]
        # now_state.append(self.get_score())
        for player in range(4):
            now_state.append([[1] if j<34 and self.score[player]&(1<<j) else [0] for j in range(136)])
        now_state.append([[1] if i==self.combo else [0] for i in range(136)])
        now_state.append([[1] if i==self.reach_b else [0] for i in range(136)])
        now_state.append([[1] if i==self.time else [0] for i in range(136)])
        now_state.append([[1] if i==self.seat else [0] for i in range(136)])
        now_state.append([[1] if i==self.round else [0] for i in range(136)])
        now_state.append([[1] if i==self.now_player_id else [0] for i in range(136)])
        now_state.append(self.wind[0])
        now_state.append(self.dora[0])
        
        for player in range(4):
            now_state.append(self.closehand[player][0])
            for i in range(8):
                now_state.append(self.openhand[player][i])
            for i in range(34):
                now_state.append(self.river[player][i])
            now_state.append(self.reach[player][0])
            now_state.append(self.selfwind[player][0])
            # gs=self.get_gain_speed(player)
            # for i in range(36):
                # now_state.append(gs[i])
        now_state.append([[1] if i==self.mountion else [0] for i in range(136)])
        now_state.append(self.mountion_detail[0])
        now_state.append(self.see_all[0])
        now_state.append([[1] if i==self.last_card else [0] for i in range(136)])
        now_state=np.array(now_state)
        # print("shape:",now_state.shape)
        return now_state
    
    def check_can_ron(self,player,who_discard):
        now_closehand=[1 if self.closehand[player][0][i][0]==1 else 0 for i in range(136)]
        now_close_cnt=0
        for i in range(136):
            if now_closehand[i]:
                now_close_cnt+=1
        now_openhand=self.openhand[player]
        now_open_cnt=self.open_cnt[player]
        now_reach=0
        for i in range(136):
            if self.reach[player][0][i][0]==1:
                now_reach=1
        now_wind=self.wind[0]
        now_selfwind=self.selfwind[player][0]
        now_card=self.last_card
        if player!=who_discard :
            now_closehand[now_card]=1
        #import mahjong
        closehand=[i for i in range(136) if now_closehand[i]==1]
        melds=get_melds(now_openhand,now_open_cnt)
        for m in melds:
            for tile in m.tiles:
                closehand.append(tile)
        now_dora=self.dora[0]
        dora=[i for i in range(136) if now_dora[i][0]==1]
        wind=[i for i in range(136) if now_wind[i][0]==1][0]//4
        selfwind=[i for i in range(136) if now_selfwind[i][0]==1][0]//4
        calculator = HandCalculator()
        result = calculator.estimate_hand_value(closehand,now_card,melds,dora,HandConfig(is_tsumo=(player==who_discard),is_riichi=(now_reach==1) ,player_wind=selfwind,round_wind=wind,options=tenhou_option))
        return not (result.cost is None)
    
    def get_score_d(self,player,who_discard):
        now_closehand=[1 if self.closehand[player][0][i][0]==1 else 0 for i in range(136)]
        now_close_cnt=0
        for i in range(136):
            if now_closehand[i]:
                now_close_cnt+=1
        now_openhand=self.openhand[player]
        now_open_cnt=self.open_cnt[player]
        now_reach=0
        for i in range(136):
            if self.reach[player][0][i][0]==1:
                now_reach=1
        now_wind=self.wind[0]
        now_selfwind=self.selfwind[player][0]
        now_card=self.last_card
        if player!=who_discard :
            now_closehand[now_card]=1
        #import mahjong
        closehand=[i for i in range(136) if now_closehand[i]==1]
        melds=get_melds(now_openhand,now_open_cnt)
        for m in melds:
            for tile in m.tiles:
                closehand.append(tile)
        now_dora=self.dora[0]
        dora=[i for i in range(136) if now_dora[i][0]==1]
        wind=[i for i in range(136) if now_wind[i][0]==1][0]//4
        selfwind=[i for i in range(136) if now_selfwind[i][0]==1][0]//4
        calculator = HandCalculator()
        result = calculator.estimate_hand_value(closehand,now_card,melds,dora,HandConfig(is_tsumo=(player==who_discard),is_riichi=(now_reach==1) ,player_wind=selfwind,round_wind=wind,options=tenhou_option))
        score_d=[0,0,0,0]
        if result.cost is None:
            return score_d
        else :
            self.reach_b=0
            wd=wind-27-player
            if wd<0:
                wd+=4
            if player==who_discard:
                for i in range(4):
                    if i==player:
                        score_d[i]=result.cost['total']//100
                    else :
                        if (wd+player)%4==0:
                            score_d[i]=-result.cost['main_bonus']//100
                        else :
                            score_d[i]=-result.cost['additional_bonus']//100
            else:
                score_d[player]=result.cost['total']//100
                score_d[who_discard]=-result.cost['main_bonus']//100
    
    def check_can_reach(self,player):
        if self.open_cnt[player]>0 :
            return 0
        now_reach=self.reach[player][0]
        for i in range(136):
            if now_reach[i][0]:
                return 0
        now_closehand=[i for i in range(136) if self.closehand[player][0][i][0]==1]
        #import mahjong
        shanten = Shanten()
        tiles = TilesConverter.to_34_array(now_closehand)
        result = shanten.calculate_shanten(tiles)
        return result<=0
    
    def get_legal_actions(self):
        legal_actions=[]
        if self.now_player_opt==0 or self.now_player_opt==1 :
            for i in range(136):
                if self.mountion_detail[0][i][0]==1 :
                    legal_actions.append(i)
        if self.now_player_opt==2:
            if self.see_opponent==1 or self.now_player_id==0 :
                for i in range(136):
                    if self.closehand[self.now_player_id][0][i][0]==1 :
                        legal_actions.append(i)
            else :
                for i in range(136):
                    if self.mountion_detail[0][i][0]==1 :
                        legal_actions.append(i)
        if self.now_player_opt==3:
            if self.now_player_opt_state==0 or self.now_player_opt_state==4:
                legal_actions.append(0)
                if self.now_player_id==0 or self.see_opponent==1 :
                    if self.check_can_ron(self.now_player_id,self.now_player_id) :
                        legal_actions.append(1<<self.now_player_id)
                else :
                    legal_actions.append(1<<self.now_player_id)
            else :
                for i in range(16):
                    if i&(1<<self.now_player_id) :
                        pass
                    else :
                        flag=1
                        for j in range(4):
                            if i&(1<<j) and (j==0 or self.see_opponent==1):
                                flag &= self.check_can_ron(j,self.now_player_id)
                        if flag:
                            legal_actions.append(i)
                        
        if self.now_player_opt==4:
            legal_actions.append(0)
            card_type=self.last_card//4
            if self.now_player_opt_state==1 or self.now_player_opt_state==3:
                for i in range(3):
                    next_player=(self.now_player_id+i+1)%4
                    if next_player==0 or self.see_opponent==1 :
                        cnt=0
                        for j in range(4):
                            if self.closehand[next_player][0][card_type*4+j][0]==1:
                                cnt+=1
                        if cnt==3:
                            legal_actions.append(i+2)
                    else :
                        cnt=0
                        for j in range(4):
                            if self.see_all[0][card_type*4+j][0]==0:
                                cnt+=1
                        if cnt==3 and self.open_cnt[next_player]<4 :
                            legal_actions.append(i+2)
            else :
                if self.now_player_id==0 or self.see_opponent==1 :
                    cnt=0
                    for j in range(4):
                        if self.closehand[self.now_player_id][0][card_type*4+j][0]==1:
                            cnt+=1
                    flag=0
                    for j in range(4):
                        cc=0
                        for k in range(4):
                            if self.openhand[self.now_player_id][j*2][card_type*4+j][0]==1:
                                cc+=1
                        if cc==3 :
                            flag=1
                    if cnt==4 or flag:
                        legal_actions.append(1)
                else :
                    cnt=0
                    for j in range(4):
                        if self.see_all[0][card_type*4+j][0]==0:
                            cnt+=1
                    flag=0
                    for j in range(4):
                        cc=0
                        for k in range(4):
                            if self.openhand[self.now_player_id][j*2][card_type*4+j][0]==1:
                                cc+=1
                        if cc==3 :
                            flag=1
                    if (cnt==4 and self.open_cnt[next_player]<4) or (flag and self.open_cnt[next_player]<=4 ):
                        legal_actions.append(1)
        if self.now_player_opt==5:
            legal_actions.append(0)
            card_type=self.last_card//4
            card_id=self.last_card%4
            d1=[1,0,0,0]
            d2=[2,2,1,1]
            d3=[3,3,3,2]
            for i in range(3):
                next_player=(self.now_player_id+i+1)%4
                if next_player==0 or self.see_opponent==1 :
                    for j in range(4):
                        if j==card_id :
                            continue
                        card1=card_type*4+d1[j]
                        card2=card_type*4+d2[j]
                        card3=card_type*4+d3[j]
                        if card1==self.last_card :
                            card1=card3
                        if card2==self.last_card :
                            card2=card3
                        if self.closehand[next_player][0][card1][0]==1 and self.closehand[next_player][0][card2][0]==1:
                            legal_actions.append(1+i*4+j)
                else :
                    for j in range(4):
                        if j==card_id :
                            continue
                        card1=card_type*4+d1[j]
                        card2=card_type*4+d2[j]
                        card3=card_type*4+d3[j]
                        if card1==self.last_card :
                            card1=card3
                        if card2==self.last_card :
                            card2=card3
                        if self.see_all[0][card1][0]==0 and self.see_all[0][card2][0]==0  and self.open_cnt[next_player]<4 :
                            legal_actions.append(1+i*4+j)
            
        if self.now_player_opt==6:    
            legal_actions.append(0)
            next_player=(self.now_player_id+1)%4
            card_type=self.last_card//4
            if tile_type(card_type)!=3:
                card1_d=[1,-1,-2]
                card2_d=[2,1,-1]
                if next_player==0 or self.see_opponent==1 :
                    for i in range(3):
                        card1=card_type+card1_d[i]
                        card2=card_type+card2_d[i]
                        if card1<0 or card1>=34 or card2<0 or card2>=34:
                            continue
                        if tile_type(card1)!=tile_type(card_type) or tile_type(card2)!=tile_type(card_type):
                            continue
                        for j in range(4):
                            for k in range(4):
                                if self.closehand[next_player][0][card1*4+j][0]==1 and self.closehand[next_player][0][card2*4+k][0]==1 and self.open_cnt[next_player]<4 :
                                    legal_actions.append(1+i*4*4+j*4+k)
                else :
                    for i in range(3):
                        card1=card_type+card1_d[i]
                        card2=card_type+card2_d[i]
                        if card1<0 or card1>=34 or card2<0 or card2>=34:
                            continue
                        if tile_type(card1)!=tile_type(card_type) or tile_type(card2)!=tile_type(card_type):
                            continue
                        for j in range(4):
                            for k in range(4):
                                if self.see_all[0][card1*4+j][0]==0 and self.see_all[0][card2*4+k][0]==0 and self.open_cnt[next_player]<4 :
                                    legal_actions.append(1+i*4*4+j*4+k)
                            
        if self.now_player_opt==7:
            legal_actions.append(0)
            if self.now_player_id==0 or self.see_opponent==1 :
                if self.check_can_reach(self.now_player_id):
                    legal_actions.append(1)
            else :
                legal_actions.append(1)
        
        return legal_actions
    def current_state(self):       
        return self.get_state_340_136_1(),self.opt_name[self.now_player_opt],self.get_legal_actions()
    
class MCTSPlayer(object):
    def __init__(self, policy_value_function, n_playout,c_puct=5,is_selfplay=0):
        self._is_selfplay = is_selfplay
        self._n_playout=n_playout
        self._c_puct=c_puct
        self._policy_value_function=policy_value_function
        
    def set_game_state(self,round_init_message):
        self.game_state=Game_State(round_init_message)
        self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout, self.game_state.get_mct_state())
        
    def get_action(self, temp=1e-3):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        mp,npi,nop,nops=self.game_state.get_mct_state()
        if nop==4 and npi!=0 :
            # 鸣牌
            move=np.array([0,0,0])
            move_probs=np.array([np.zeros(5),np.zeros(13),np.zeros(49)])
            acts, probs = self.mcts.get_move_probs(self.game_state, temp)
            move_probs[0][list(acts)] = probs
            if self._is_selfplay:
                _p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                move[0] = np.random.choice(acts,p=_p)
            else:
                move[0] = np.random.choice(acts,p=_p)
            if move[0]>0 and (nop+move[0]-1)%4!=mp :
                move[0]=0
            if move[0]==0 :
                acts, probs = self.mcts.get_move_probs_pong(temp)
                move_probs[1][list(acts)] = probs
                if self._is_selfplay:
                    _p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                    move[1] = np.random.choice(acts,p=_p)
                else:
                    move[1] = np.random.choice(acts,p=_p)
                if move[1]>0 and (npi+((move[1]-1)//4))%4!=mp :
                    move[1]=0
                if move[1]==0 and (npi+1)%4==mp :
                    acts, probs = self.mcts.get_move_probs_chow(temp)
                    move_probs[2][list(acts)] = probs
                    if self._is_selfplay:
                        _p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                        move[2] = np.random.choice(acts,p=_p)
                    else:
                        move[2] = np.random.choice(acts,p=_p)
            return move
        
        else :
            # 自己回合的kong/reach/discard 和 全部ron
            move_probs = np.zeros(self.game_state.get_num_move())
            acts, probs = self.mcts.get_move_probs(self.game_state, temp)
            move_probs[list(acts)] = probs
            if nop==3 :
                cnt=0
                for i in acts:
                    if i&(1<<mp):
                        cnt+=1
                if cnt==0:
                    _p=np.array([1])
                    acts=np.array([0])
                else:
                    tmp=np.array([0,0])
                    for i in range(16):
                        if i&(1<<mp):
                            tmp[0]+=move_probs[i]
                        else:
                            tmp[1]+=move_probs[i]
                    _p=tmp
                    acts=np.array([0,1])
            else:
                _p=probs
            if self._is_selfplay:
                move = np.random.choice(acts,p=0.75*_p + 0.25*np.random.dirichlet(0.3*np.ones(len(_p))))
            else:
                move = np.random.choice(acts,p=_p)
            return move
    
    def get_move_probs(self, temp=1e-3):
        move_probs = np.zeros(self.game_state.get_num_move())
        acts, probs = self.mcts.get_move_probs(self.game_state, temp)
        move_probs[list(acts)] = probs
        return move_probs
                                  
    def do_action(self,action,close_kong=0,extra_date=None):
        if self._is_selfplay :
            self.mcts.update_with_move(action,close_kong)
            self.game_state.do_action(action,close_kong,extra_date)
        else :
            self.game_state.do_action(action,close_kong,extra_date)
            self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout, self.game_state.get_mct_state())

    def __str__(self):
        return "MCTS {}".format(self.player)
