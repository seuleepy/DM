import numpy as np
import pandas as pd

infile = "/home/sgjeong/workspace/ML/binary_10.h5"

df = pd.read_hdf(infile)
print("df:\n",df)
print("\n columns:", df.columns)
