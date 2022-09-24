#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import awkward as ak
import myModule
import sklearn
import tables


# In[2]:


def dfMaker(inputset, features):
    
    count = 0
    
    for process in set:
        columns = features
        nb_evt = len(inputset[process]['MET_PT'])
        
        if process == "signal":
            y = np.ones(nb_evt)
        else:
            y = np.zeros(nb_evt)
            
        gen_evt = genEventDict[process]
        xsec = xsecDict[process]
            
        data = {'y' : y, 'Gen Event' : gen_evt, 'xsec' : xsec}
        df_ps = pd.DataFrame(data)
            
        for column in columns:
            
            if nb_evt == len(ak.flatten(inputset[process][column])):
                df_ps[column] = ak.to_pandas(ak.flatten(inputset[process][column]))
            else:
                i = []
                j = []
                k = []
                for idx_evt in range(nb_evt):
                    i.append(inputset[process][column][idx_evt][0])
                    j.append(inputset[process][column][idx_evt][1])
                    k.append(inputset[process][column][idx_evt][2])
                df_ps["1 " + column] = pd.DataFrame(i)
                df_ps["2 " + column] = pd.DataFrame(j)
                df_ps["3 " + column] = pd.DataFrame(k)
                
        if count == 0:
            df = df_ps
            count += 1
        else:
            df = pd.concat([df, df_ps])
            
    return df


# In[3]:


filepath = "/home/sgjeong/workspace/ttb_preselection/ttb_ml_ntuple"


mclist = ["signal01",
        "1l",
	"1l_2",
	"2l",
	"2l_2",
	"FH",
	"FH_2",
	"TTW",
	"TTZ",
	"ZZ",
	"WZ",
	"WW"]


# In[4]:


mc = myModule.npyLoader(filepath, mclist)

features = []

for d in mc[0].keys():
    features.append(d)


# In[5]:
def sum_dict(a, b):
    dict_ = {}
    for ft in features:
        dict_[ft] = np.concatenate([a[ft], b[ft]])
    return dict_

semi = sum_dict(mc[1], mc[2])
di = sum_dict(mc[3], mc[4])
FH = sum_dict(mc[5], mc[6])

set = {
	'signal' : mc[0],
	'1l' : semi,
	'2l' : di,
	'FH' : FH,
	'TTW' : mc[7],
	'TTZ' : mc[8],
	'ZZ' : mc[9],
	'WZ' : mc[10],
	'WW' : mc[11]
}


# In[6]:


lumi = 3000000


# In[7]:


genEventDict = {
	'signal' : 100000,
	'1l' : 11810000,
	'2l' : 10100000,
	'FH' : 12000000,
	'TTW' : 10000000,
	'TTZ' : 10000000,
	'ZZ' : 10000000,
	'WZ' : 10000000,
	'WW' : 10000000
}


# In[8]:


xsecDict = {
	'signal' : 2.532e-01,
	'1l' : 211.1,
	'2l' : 40.288,
	'FH' : 210.534,
	'TTW' : 0.3664,
	'TTZ' : 0.6428,
	'ZZ' : 9.841,
	'WZ' : 26.16,
	'WW' : 68.3
}







df = dfMaker(set,features)


# In[12]:


#df


# In[13]:


pd.set_option("display.max_colwidth", 200)


# In[14]:


df['weight'] = (df['xsec'] * lumi) / df['Gen Event']



sigN = len(df['weight'][df['y'] == 1]) # signal의 개수


# In[21]:


bkgN = (df['weight'][df['y'] == 0]).sum() # sigma(weight * nb_evt)


# In[22]:


SF = sigN/bkgN


# In[23]:


print('Signal : {} BKG : {} SF : {}'.format(sigN, bkgN, SF))


# In[24]:


df['weight'][df['y'] == 1] = 1
df['weight'][df['y'] == 0] = df['weight'][df['y'] == 0] * SF


# In[25]:


#df


# In[26]:


df_shuf = sklearn.utils.shuffle(df)


# In[27]:


df_shuf.to_hdf('binary_01.h5', key = 'df', mode = 'w')

