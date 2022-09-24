import csv


lumi = 3000000


# In[7]:


genEvent = [100000, 11810000, 10100000,12000000,10000000, 10000000,10000000,10000000, 10000000]

set = ['sig', '1l', '2l', 'FH', 'TTW', 'TTZ', 'ZZ', 'WZ', 'WW']
# In[8]:


xsec = [2.188e-04, 211.1, 40.288, 210.534,0.3664, 0.6428,9.841,26.16, 68.3]

f = open("df_weight.csv", 'w')
writer = csv.writer(f)

data = [['set', 'weight']]

for i in range(len(genEvent)):
	weight = xsec[i] * lumi / genEvent[i]
	data.append([set[i], weight])
	print(set[i], weight, '\n')

writer.writerows(data)
f.close()
