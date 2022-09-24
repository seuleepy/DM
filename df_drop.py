import numpy as np
import pandas as pd

infile = "/home/sgjeong/workspace/ML/binary_01.h5"

df = pd.read_hdf(infile)
#print("df:\n",df)
#print("\n columns:", df.columns)

#print(type(df))

df = df.drop(labels = ['1 Jet_PT', '2 Jet_PT', '3 Jet_PT','1 Jet_Eta', '2 Jet_Eta', '3 Jet_Eta', '1 Jet_Phi', '2 Jet_Phi', '3 Jet_Phi',  '1 Jet_Mass', '2 Jet_Mass', '3 Jet_Mass'], axis = 1)
print(df)

df.to_hdf('binary_01_0816.h5', key = 'df', mode = 'w')
