#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class Model(nn.Module):
	def __init__(self, input_node):
		self.input_node = input_node
        
		super(Model, self).__init__()
        
		self.fc = nn.Sequential(
            
		nn.Linear(input_node, 32),
		nn.ReLU(),
		nn.BatchNorm1d(32),

		nn.Linear(32, 16),
		nn.ReLU(),
		nn.BatchNorm1d(16),
	
		nn.Linear(16, 16),
		nn.ReLU(),
		nn.BatchNorm1d(16),

		nn.Linear(16, 4),
		nn.ReLU(),
		nn.BatchNorm1d(4),

		nn.Linear(4, 1),
		nn.Sigmoid()
	)

	def forward(self, x):
		y_pred = self.fc(x)
		return y_pred

