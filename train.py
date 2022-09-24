#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys, os
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch.utils.data.dataset import random_split
import torch.nn.functional as F


# In[2]:


## Check GPU

def GPU_check():
    
    if torch.cuda.is_available():
        nGPU = torch.cuda.device_count()
        print("Number of GPU", nGPU)
        
        for i, j in enumerate(range(nGPU)):
            print("Device", i, torch.cuda.get_device_name(i))
    
    else:
        print("No GPU for use")


# In[3]:


GPU_check()


# In[4]:


use_GPU = True


# In[5]:


import argparse


# In[6]:



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type = int, default = 200,
                   help = "--epoch EPOCH")
parser.add_argument('--batch', type = int, default = 4096,
                   help = "--batch BATCH_SIZE")
parser.add_argument('--lr', type = float, default = 0.001,
                   help = "--lr LEARNING_RATE")

args = parser.parse_args()



# In[7]:



## Hyperparameter

batch_size = args.batch
LR = args.lr
EPOCH = args.epoch


#batch_size = 4096
#LR = 0.001
#EPOCH = 200


# In[8]:


## Data load

sys.path.append("../python/")

from DataLoader import TrainDataset, ValDataset

train_dataset = TrainDataset()
val_dataset = ValDataset()

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size * 2, shuffle = False, num_workers = 2)


# In[12]:


import myModule
input_node = myModule.take_input_node(train_loader)


# In[15]:


from Model import Model

if torch.cuda.is_available() & use_GPU:
    device = 'cuda'
else:
    device = 'cpu'
model = Model(input_node)
model.to(device)
model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
model.cuda()


# In[22]:


optim = optim.Adagrad(model.parameters(), lr = LR)


# In[23]:


from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
# In[17]:


'''
for i, (a, b, c) in enumerate(train_loader):
    train_x = a.float().to(device)
    label = b.float().to(device)
    weight = c.float().to(device)
            
    pred = model(train_x)
    loss = nn.BCELoss(weight = weight)
    print(weight)
    print(weight.shape)
    print(len(train_loader))
    break
'''


# In[32]:

lr_scheduler = ReduceLROnPlateau(optim, mode = 'min', factor = 0.5, verbose = 1, cooldown = 10)
#lr_scheduler = CosineAnnealingLR(optim, T_max = 50, eta_min = 0.00001, verbose = 1)
bestWeight, bestLoss = {}, 1e9

try:
    
	history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
	## EPOCH
	for epoch in tqdm(range(1, EPOCH + 1)):
        
		# Training Stage
		model.train()
		optim.zero_grad() # initialize grad data
		train_loss, train_acc = 0., 0.
        
		for i, (train_x, label, train_w) in enumerate(train_loader):
            
			train_x = train_x.float().to(device) # Make tensor on the gpu or cpu # .float() : float64 -> float32
			label = label.float().to(device)
			weight = train_w.float().to(device)
            
			pred = model(train_x)
			crit = nn.BCELoss(weight = weight) # binary cross entropy # weight option ??
           
			if device == 'cuda': crit = crit.cuda()
			loss = crit(pred, label)
			loss.backward()
            
			optim.step()
			train_loss += loss.item()
			train_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0), sample_weight = weight.view(-1).to('cpu'))
        
		train_loss /= len(train_loader)
		train_acc /= len(train_loader)
        
		# Validation Stage
		model.eval()
		val_loss, val_acc = 0., 0.
	
		for i, (val_x, label, val_w) in enumerate(val_loader):
			val_x = val_x.float().to(device)
			label = label.float().to(device)
			weight = val_w.float().to(device)
            
			pred = model(val_x)
			crit = nn.BCELoss(weight = weight)
            
			loss = crit(pred, label)
			val_loss += loss.item()
			val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0), sample_weight = weight.view(-1).to('cpu'))
            
		val_loss /= len(val_loader)
		val_acc /= len(val_loader)
       
		lr_scheduler.step(loss)	
 
		# Update weight of best epoch checking validation loss
		# validation loss가 떨어질때만 저장. 과적합 방지.
		if bestLoss > val_loss:
			bestWeight = model.state_dict() # weight, bias 정보
			bestlLoss = val_loss
			torch.save(bestWeight, 'weightFile_01_01.pth')
            
		history['train_loss'].append(train_loss)
		history['train_accuracy'].append(train_acc)
		history['val_loss'].append(val_loss)
		history['val_accuracy'].append(val_acc)
		#if epoch % 10 == 0:
		if True:
			print("Epoch: {0}, Train Loss:{1}, Val Loss :{2}, Train Acc: {3}, Val Acc: {4}".format(epoch, train_loss, val_loss, train_acc, val_acc))
		with open('history_01_01.csv', 'w') as f:
			writer = csv.writer(f)
			keys = history.keys() # type : dict_keys
			writer.writerow(keys)
			for row in zip(*[history[key] for key in keys]):
				writer.writerow(row)
except KeyboardInterrupt:
	print("try again")


# In[48]:


'''
for i, (val_x, label, val_w) in enumerate(val_loader):
    val_x = val_x.float().to(device)
    label = label.float().to(device)
    val_w = val_w.float().to(device)
    #print(val_x.shape)
    print(label.shape)
    pred = model(val_x)
    print(pred.shape)
    #print(val_w.shape)
    break
'''


# In[ ]:


'''
for i, (train_x, label, train_w) in enumerate(train_loader):
    
    train_x = train_x.float().to(device) # Make tensor on the gpu or cpu # .float() : float64 -> float32
    label = label.float().to(device)
    weight = train_w.float().to(device)
    
    pred = model(train_x)
    print(pred.shape)
    print(label.shape)
    #print(train_x.shape)
    break
'''

