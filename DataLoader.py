#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import from_numpy
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# In[3]:


# Dataset class

class TrainDataset(Dataset):
    def __init__(self):
        self.train_x = from_numpy(train_x)
        self.train_y = from_numpy(train_y)
        self.train_w = from_numpy(train_w)
        
    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index], self.train_w[index]
    
    def __len__(self):
        self.len = len(train_df)
        return self.len


# In[4]:


class ValDataset(Dataset):
    def __init__(self):
        self.val_x = from_numpy(val_x)
        self.val_y = from_numpy(val_y)
        self.val_w = from_numpy(val_w)
        
    def __getitem__(self, index):
        return self.val_x[index], self.val_y[index], self.val_w[index]
    
    def __len__(self):
        self.len = len(val_df)
        return self.len


# In[5]:


class TestDataset(Dataset):
    def __init__(self):
        self.test_x = from_numpy(test_x)
        self.test_y = from_numpy(test_y)
        self.test_w = from_numpy(test_w)
        
    def __getitem__(self, index):
        return self.test_x[index], self.test_y[index], self.test_w[index]
    
    def __len__(self):
        self.len = len(test_df)
        return self.len


# In[6]:


infile = "./binary_09.h5"

df = pd.read_hdf(infile)


# In[7]:


train_df, val_df, test_df = np.split(df.sample(frac = 1, random_state = 42),
                                    [int(.8*len(df)), int(.9*len(df))])



# In[9]:


# Train data

train_x = train_df.iloc[:, 3:-1].values # sereis type -> 1차배열 event 정보
train_y = train_df.iloc[:, [0]].values # DataFrame type -> 2차벡터 y 정보
train_w = train_df.iloc[:, [-1]].values # DataFrame type -> 2차벡터 weight 정보


# In[10]:


print("train_x shape :", train_x.shape, "\ntrain_y shape:", train_y.shape, "\ntrain_w shape:", train_w.shape)


# In[9]:


# Validation data

val_x = val_df.iloc[: , 3:-1].values
val_y = val_df.iloc[:, [0]].values
val_w = val_df.iloc[:, [-1]].values

print("val_x shape:", val_x.shape, "\nval_y shape:", val_y.shape, "\nval_w shape:", val_w.shape)
# In[10]:


test_df = test_df.reset_index(drop = True)
# shuffle되어있던 index를 0부터 ~ index 재배열 drop True로 기존의 index를 column에 추가하지 않기

test_x = test_df.iloc[:, 3:-1].values
test_y = test_df.iloc[:, [0]].values
test_w = test_df.iloc[:, [-1]].values

print("test_x shape:", test_x.shape, "\ntest_y shape:", test_y.shape, "\ntest_w shape:", test_w.shape)
# In[1]:


std_scl = StandardScaler() # 평균0 분산1로 조정
std_scl.fit(train_x) # train_x의 분포를 객체에 저장

base_scl = MinMaxScaler() # (X-Xmin)/(Xmax-Xmin), 모든 값을 0과 1사이의 값으로 조정
base_scl.fit(train_x)


# In[2]:

MET_df = train_x[:, 0]

print(MET_df, len(MET_df))
plt.figure(figsize = (10, 6))
plt.hist(MET_df, bins = 100, range = (0, 500))
plt.xlabel("MET")
plt.ylabel("Number of Events")
plt.yscale('log')
plt.savefig("MET_01")

train_x = std_scl.transform(train_x)
val_x = std_scl.transform(val_x)
test_x = std_scl.transform(test_x)
MET_std = train_x[:, 0]

print(MET_std, len(MET_std))
plt.figure(figsize = (10, 6))
plt.hist(MET_std, bins = 100)
plt.xlabel("MET")
plt.ylabel("Number of Events")
plt.yscale('log')
plt.savefig("MET_std_01.png")

#print(max(MET_std))


# In[11]:


test_df.to_hdf('testset_09.h5', key = 'df', mode = 'w')
