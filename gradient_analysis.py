# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:00:08 2024

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

def load_variable_from_file(filename):
    with open(filename, 'rb') as file:
        variable = pickle.load(file)
    return variable

def save_variable(data, name):
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(data, file)
        
        
grad0 = load_variable_from_file('0_gradient.pkl')
grad20 = load_variable_from_file('20_gradient.pkl')