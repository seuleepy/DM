#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch.utils.data.dataset import random_split
import torch.nn.functional as F


# In[12]:


# check GPU

def GPU_check():
    
    if torch.cuda.is_available():
        
        nGPU = torch.cuda.device_count()
        print("Number of GPU :", nGPU)
        
        for i, j in enumerate(range(nGPU)):
            print("Device", i, torch.cuda.get_device_name(i))
            
    else:
        print("No GPU for use")


# In[13]:


GPU_check()


# In[14]:


use_GPU = True


# In[15]:



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type = int, default = 125,
                   help = "--epoch EPOCH")
parser.add_argument('--batch', type = int, default = 2098,
                   help = "--batch BATCH_SIZE")
parser.add_argument('--lr', type = float, default = 0.001,
                   help = "--lr LEARNING_RATE")
args = parser.parse_args()



# In[16]:


# Hyperparameter

#batch_size = args.batch
#LR = args.lr
#EPOCH = args.epoch

batch_size = 4096
#LR = 0.01
#EPOCH = 1


# In[17]:


#sys.path.append("../python")


# In[18]:


from DataLoader import TestDataset

test_dataset = TestDataset()
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)


# In[22]:


import myModule
input_node = myModule.take_input_node(test_loader)


# In[23]:


#input_node = 17


# In[25]:


# Device set and Optimizer set

from Model import Model

if torch.cuda.is_available() & use_GPU:
    device = 'cuda'
else:
    device = 'cpu'

model = Model(input_node)
model = model.to(device)

model.load_state_dict(torch.load('weightFile_10_00.pth'), strict = False) # 저장된 weight와 bias (parameter) 불러오기
#optim = optim.Adam(model.parameters(), lr = LR)


# In[27]:


# Evaluation

from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

labels, preds = [], []
weights, scaleWeights = [], []
predFile = 'prediction_09_01_wght10.csv'

for i, (test_x, label, test_w) in enumerate(tqdm(test_loader)):
    
    test_x = test_x.float().to(device)
    test_w = test_w.float()
    pred = model(test_x).detach().to('cpu').float()
    
    labels.extend([x.item() for x in label]) # extend : iterable의 각 항목을 추가
    preds.extend([x.item() for x in pred.view(-1)])
    weights.extend([x.item() for x in test_w.view(-1)])
    
df = pd.DataFrame({'label': labels, 'prediction': preds, 'weight': weights})
df.to_csv(predFile, index = True)

