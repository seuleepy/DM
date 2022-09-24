#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[6]:


def npyLoader(directory, tuplelist):
    outputlist = []
    for f in tuplelist:
        outputlist.append(np.load(directory+"/"+str(f)+"_nTuple.npy", allow_pickle = True)[()])
    outputlist = np.array(outputlist)
    return outputlist


# In[1]:


def take_input_node(loader):
    for i, (x, label, w) in enumerate(loader):
        break
    return x.shape[1]

