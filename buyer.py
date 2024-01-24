# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:48:54 2024

@author: youjingyi
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



GAMMA = 0.99

def load_variable_from_file(filename):
    with open(filename, 'rb') as file:
        variable = pickle.load(file)
    return variable


class decision(nn.Module):
    def __init__(self, win_size):
        super(decision, self).__init__()
        self.model = nn.Sequential(nn.Linear(win_size, 1024), nn.Dropout(0.5), nn.ReLU(),nn.Linear(1024, 3), nn.Softmax(dim = -1))
        self.initialize()
        
    def sample(self, prob):
        return torch.multinomial(prob, num_samples=1, replacement=True)
        
    def forward(self, states):
        
        prob = self.model(states)
        
        
        return self.sample(prob), prob
    
    
    def initialize(self):
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
                    
# 0: buy
# 1: hold
# 2: sell

class buyer():
    def __init__(self, file, win_size):
        
        
        self.max_steps = 1000
        self.init_cur = 100000
        self.price = load_variable_from_file(file)['sz.002049']['close'].tolist()
        self.test_data = load_variable_from_file(file)['sz.002049']['close'].tolist()[-1000:]
        # pdb.set_trace()
        self.test_state = load_variable_from_file(file)['sz.002049']['close'][-1000:].pct_change(periods=1).dropna().tolist()
        
        self.row_data = load_variable_from_file(file)['sz.002049']['close'].pct_change(periods=1).dropna().tolist()
        # print(self.row_data)
        self.win_size = win_size
        self.agent = decision(self.win_size)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True)
        # print(self.states)
        # self.store_data = 
        # self.reward_state = 
    def get_color_and_marker(self, decision):
        if decision == 0:
            return "r", "^" # 红色正三角形
        # elif decision == 1:
        #     return "b", "o" # 蓝色圆点
        elif decision == 2:
            return "g", "v" # 绿色倒三角形
        else:
            return "k", "." # 黑色点       
        
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
        plt.plot(data)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        
        # 显示图形
        # plt.show()
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
        record = []
        if d == 0 and self.left_cur >= 100 * p:
            self.left_cur -= 100 * p
            self.left_stock += 100
        elif d == 2 and self.left_stock > 0:        
            self.left_cur += 100 * p
            self.left_stock -= 100
        # print(self.left_cur)
        # print(self.left_stock)
        # print('money:', self.left_cur + self.left_stock * p - self.init_cur)
        return (self.left_cur + self.left_stock * p - self.init_cur) / self.init_cur
        
    def get_training_data(self, batch_size):
        sampled_data = random.sample(self.buffer, batch_size)
        decision = torch.stack([torch.tensor(item[0]) for item in sampled_data])
        next_state = torch.stack([item[3] for item in sampled_data])
        q_value = torch.stack([item[2] for item in sampled_data])
        reward = torch.stack([torch.tensor(item[4]) for item in sampled_data])
        return decision, q_value, next_state, reward
    
    def __call__(self):
         loss_array = []
         test_reward_record = []
         self.agent.train()
         for epoch in tqdm(range(1000)):
             
             self.buffer = [] 
             for batch in range(4):
                 self.left_cur = self.init_cur
                 self.left_stock = 0
                 self.today = np.random.randint(self.win_size, len(self.row_data) - 1000)
                          
                 
                 for step in range(self.max_steps):
                     today_price = self.price[self.today - 1]
                     today_state = torch.tensor(self.row_data[self.today - self.win_size:self.today])
                     # self.agent.eval()
                     cur_decision, q_value = self.agent(today_state)
                     ## [state, q_value, next_state, reward]
                     reward = self.reward_cal(cur_decision, today_price)
                     
                     self.buffer.append([cur_decision, today_state, q_value,\
                                    torch.tensor(self.row_data[self.today - self.win_size + 1:self.today + 1]),\
                                        reward])
                         
                     self.today += 1
            
                     
             ###### train ######
             # pdb.set_trace()
             t_decisions, t_q_value, t_next_state, t_reward = self.get_training_data(64)
             # print(training_data)
             # exit()
             _, next_q_value = self.agent(t_next_state)
             
             target = t_reward + GAMMA * torch.max(next_q_value, dim = -1)[0]
             t_q_value = torch.gather(t_q_value, dim = -1, index = t_decisions)[:, 0]
             loss = F.mse_loss(t_q_value, target)
             self.optimizer.zero_grad()
             loss.backward()
             self.optimizer.step()
             loss_array.append(loss.item())
             # print(loss_array)
              # print(loss_array)
             if (epoch) % 50 == 0:
                  self.draw_fig(loss_array, 'loss_record')
             if epoch % 100 == 0:
                  decision_list = []
                  self.left_cur = self.init_cur
                  self.left_stock = 0
                  self.agent.eval()
                  count = 0
                  for  step in range(self.win_size - 1, len(self.test_state)):
                      # pdb.set_trace()
                      test_state_inp = torch.tensor(self.test_state[step - self.win_size + 1 : step + 1])
                     
                      test_cur_decision, test_q_value = self.agent(test_state_inp)
                      
                      test_reward = self.reward_cal(test_cur_decision, self.test_data[step])
                      count += 1
                      decision_list.append(test_cur_decision)
                  test_reward_record.append(test_reward)
                  print(test_reward_record)
                  self.draw_fig(test_reward_record, 'test_reward_record')
                  self.test_draw(self.test_data[self.win_size:], decision_list, f'{epoch}_test')