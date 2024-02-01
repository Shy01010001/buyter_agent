# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:21:46 2024

@author: hongyu
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.optim as optim
import pdb
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import gradcheck

GAMMA = 0.99
np.random.seed(2024)
random.seed(2024)

def load_variable_from_file(filename):
    with open(filename, 'rb') as file:
        variable = pickle.load(file)
    return variable

def save_variable(data, name):
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(data, file)
        
        
class decision(nn.Module):
    def __init__(self, win_size, dropout = 0.5):
        super(decision, self).__init__()
        self.coach = nn.Sequential(nn.Linear(win_size, 2048), nn.Dropout(0.5), nn.ReLU(),nn.Linear(2048, 1), nn.Sigmoid())
        self.model_pos = nn.Sequential(nn.Linear(win_size, 2048), nn.Dropout(0.5), nn.ReLU(),nn.Linear(2048, 3), nn.Softmax(dim = -1))
        self.model_neg = nn.Sequential(nn.Linear(win_size, 2048), nn.Dropout(0.5), nn.ReLU(),nn.Linear(2048, 3), nn.Softmax(dim = -1))
        self.initialize()
        
    def sample(self, prob):
        
        return torch.multinomial(prob, num_samples=1, replacement=True)
        
    def forward(self, states):
        # print(states)
        # x = input()
        # if x == '0':
        #     pass
        # else:
        #     exit()
        # print(states.size())
        guide = self.coach(states)
        pos = self.model_pos(states)
        neg = self.model_neg(states)
        
        prob = guide * pos + (1 - guide) * neg
        
        return self.sample(prob), prob
    
    def initialize(self):
        for layer in self.model_pos.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
                    
        for layer in self.model_neg.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)    
                    
        for layer in self.coach.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)                         
                    
# 0: buy
# 1: hold
# 2: sell

class buyer():
    def __init__(self, file, win_size, experiment_group):
        try:
            os.mkdir('lr_record')
            os.mkdir('q_value_record')
            os.mkdir('reward_record')
            os.mkdir('test')
            os.mkdir('loss_record')
            # os.mkdir('mood_record')
            
        except:
            pass        
        self.save_file_name = experiment_group
        self.max_steps = 1000
        self.init_cur = 10000
        self.price = load_variable_from_file(file)['sz.002049']['close'].tolist()
        self.test_data = load_variable_from_file(file)['sz.002049']['close'].tolist()[-1000:]
        # pdb.set_trace()
        self.test_state = load_variable_from_file(file)['sz.002049']['close'][-1000:].pct_change(periods=1).tolist()
        self.test_volume = load_variable_from_file(file)['sz.002049']['volume'][-1000:].pct_change(periods=1).tolist()
        
        self.raw_data = load_variable_from_file(file)['sz.002049']['close'].pct_change(periods=1).tolist()
        # pdb.set_trace()
        self.raw_volume = load_variable_from_file(file)['sz.002049']['volume'].pct_change(periods=1).tolist()
        self.raw_data = self.clear_nan(self.raw_data)
        self.raw_volume = self.clear_nan(self.raw_volume)
        
        self.test_volume = self.clear_nan(self.test_volume)
        self.test_state = self.clear_nan(self.test_state)
        self.win_size = win_size
        
        self.agent = decision(self.win_size, 0.5).float()
        self.target_agent = decision(self.win_size, 0.5).float()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR( self.optimizer, 
                                                step_size = 50, 
                                                gamma = 0.1)
    def z_zero_nomalization(self, data):
        # if isinstance(data, torch.Tensor):
            # pass
        # else:
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 0.0000000001)
        
    def get_color_and_marker(self, decision):
        if decision == 0:
            return "r", "^" # 红色正三角形
        # elif decision == 1:
        #     return "b", "o" # 蓝色圆点
        elif decision == 2:
            return "g", "v" # 绿色倒三角形
        else:
            return "k", "." # 黑色点       
        
    
    
    def clear_nan(self, inp):
        mask = np.isinf(inp)
        inp = np.where(mask, 0, inp)
        mask = np.isnan(inp)
        return np.where(mask, 0, inp)
        
    def test_draw(self, data, test_decision_list, file_name):
        plt.figure()
        x_list = []
        y_list = []        
        for x, y, d in zip(range(len(data)), data, test_decision_list):
            # 获取颜色和形状
            c, m = self.get_color_and_marker(d)
            # 画出点
            plt.plot(x, y, c + m)
            # 添加数据标签
            # plt.text(x, y + 0.5, str(y), ha="center", va="bottom", fontsize=10) 
            x_list.append(x)
            y_list.append(y)

# 画出折线图
        plt.plot(x_list, y_list, "k-")            
        plt.xlabel("date")
        plt.ylabel("price/￥")
        plt.savefig(f'{file_name}.jpg')
            
    def draw_fig(self, data, file_name):
        plt.figure()
        plt.plot(data)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(f'{file_name}.jpg')
           
    def reward_cal(self, d, p):
        """
        当前步的rewards计算
    
        参数：
        - d: 当前步的决策，0，1，2
        - p: 当前price
    
        返回值：
        None
        """
        if d == 0 and self.left_cur >= 100 * p:
            self.left_cur -= 100 * p
            self.left_stock += 100
        elif d == 2 and self.left_stock > 0:        
            self.left_cur += 100 * p
            self.left_stock -= 100
        self.mood.pop(0)
        # print(self.mood)
        cur_rw = (self.left_cur + self.left_stock * p - self.init_cur) / self.init_cur
        self.mood.append(cur_rw*2 if cur_rw<0 else cur_rw)
        return self.mood[-1]
        
    def get_training_data(self, batch_size, n_step):
        sampled_data = random.sample(self.buffer, batch_size)
        decision = torch.stack([item[0] for item in sampled_data])
        next_state = torch.stack([item[3] for item in sampled_data])
        q_value = torch.stack([item[2] for item in sampled_data])
        reward = torch.stack([torch.tensor(item[4], dtype = torch.float) for item in sampled_data])
        return decision, q_value, next_state, reward
    
    def agent_perform(self, begin: int, steps: int, interval: int, raw_data: list, raw_volume: list, mode: str = 'train') -> None:
        if mode == 'train':
            self.agent.train()
        else:
            self.agent.eval()
        today_state = self.z_zero_nomalization(raw_data[begin : begin + self.win_size])
        today_state = torch.tensor(today_state, dtype = torch.float)
        
        
        today_price = self.price[begin + self.win_size]
        for step in range(1, steps):
            try:
                cur_decision, q_value = self.agent(today_state)
            except:
                print(today_state)
                print(today_state.size())
                print(begin)
                print(step)
                exit()
            reward = self.reward_cal(cur_decision, today_price)
            next_day_state = self.z_zero_nomalization(raw_data[begin + step * interval : begin + step * interval + self.win_size])
            next_day_state = torch.tensor(next_day_state, dtype = torch.float)
            
            if mode == 'train':
                self.buffer.append([cur_decision, today_state, q_value, next_day_state, reward])
            else:
                max_q_value = max(q_value)
                # print(type(max_q_value))
                self.buffer_test.append([cur_decision, today_state, q_value, next_day_state, reward, max_q_value])
            today_state = next_day_state
            today_price = self.price[begin + step * interval + self.win_size]
            # return today_state
    def __call__(self):
         # self.buffer 中四个元素的类型分别是：tensor, tensor, tensor, int
         loss_array = []
         lr_list = []
         loss_var_list = []
         # mood_record_list = []
         test_reward_record = []
         self.agent.train()
         self.target_agent.eval()
         for epoch in tqdm(range(1200)):
             
             self.buffer = []
             self.buffer_test = []
             for batch in range(4):
                 self.mood = [0 for i in range(self.win_size)]
                 self.left_cur = self.init_cur
                 self.left_stock = 0
                 self.today = np.random.randint(self.win_size, len(self.raw_data) - 1060)
                 self.agent_perform(self.today - self.win_size, self.max_steps, 1, self.raw_data, self.raw_volume) ## magic number interval
                 
             ###### train ######
             parameters = self.agent.parameters()
             
             t_decisions, t_q_value, t_next_state, t_reward = self.get_training_data(128, 10) ## batch_size, n_step_q_learning ##
             
             _, next_q_value = self.target_agent(t_next_state)
             
             target = t_reward + GAMMA * torch.max(next_q_value, dim = -1)[0]
             t_q_value = torch.gather(t_q_value, dim = -1, index = t_decisions)[:, 0]
             
             loss = F.mse_loss(t_q_value, target)
             # pdb.set_trace()
             # gradients = torch.autograd.grad(loss, self.agent.parameters(), create_graph=True)
             # save_variable(gradients, f'{epoch}_gradient')
             self.optimizer.zero_grad()
             loss.backward()
             # torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
             self.optimizer.step()
             self.lr_scheduler.step()
             lr_list.append(self.lr_scheduler.get_last_lr())
             loss_array.append(loss.item())
             # current_lr = self.optimizer.param_groups[0]['lr']
             # print()
             # print()
             # print('loss value:', loss.item())
             # print()
             # print()
             # # print(loss)
             # print('loss_var_list: ',loss_var_list)
             # print()
             # print()
             # print('lr: ',current_lr)
             # print()
             # print()
             if (epoch + 1) % 20 == 0:
                 self.target_agent.load_state_dict(self.agent.state_dict())
             if (epoch) % 1199 == 0:
                  
                  # print()
                  # print(np.array(mood_record_list).shape)
                  # self.draw_plot_fig(mood_record_list)
                  self.draw_fig(loss_array, f'./loss_record/loss_record_{self.save_file_name}')
                  self.draw_fig(loss_array, f'./lr_record/lr_record_{self.save_file_name}')
                  
             if epoch % 100 == 0:
                  # self.mood = [0] * 30
                  decision_list = []
                  self.left_cur = self.init_cur
                  self.left_stock = 0
                  
                  self.agent_perform(0, len(self.test_state) - self.win_size, 1, self.test_state, self.test_volume, mode = 'test')
                  decision_list = [item[0] for item in self.buffer_test]
                  test_reward_record = [item[-2] for item in self.buffer_test]
                  r_c = 0
                  for r in test_reward_record:
                      if r > 0:
                         r_c += 1
                  if r_c / len(test_reward_record) > 0.8:
                      gain = 'gain'
                  else:
                      gain = 'danm'
                  q_value_record_list = [item[-1].tolist() for item in self.buffer_test]
                  # print(q_value_record_list)
                  # print(type(test_reward_record))
                  # exit()
                  self.draw_fig(q_value_record_list, f'./q_value_record/{gain}_{epoch}_q_value_{self.save_file_name}')
                  self.draw_fig(test_reward_record, f'./reward_record/{gain}_{epoch}_test_reward_record_{self.save_file_name}')
                  self.test_draw(self.test_data[self.win_size:], decision_list, f'./test/{gain}_{epoch}_test_{self.save_file_name}')
                     
                 
                 

agent = buyer('./ziguang.pkl', 30, 'price_and_volume_128_a3c_price_only_aug_neg_guided_w_norm_2048')
agent()