import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

infile = "/home/sgjeong/workspace/ML/binary_10.h5"

df = pd.read_hdf(infile)
	
df_cor = df.iloc[:, 3:-1].corr()
	
print(df_cor)

for label, content in df_cor.items():
	print("label:",  label)
	print("content:")
	if content[0] >= 0.5:
		print(content)

sns.clustermap(df_cor, cmap = 'RdYlBu_r', linewidth=.5, vmin=-1, vmax=1)	
plt.savefig("cor_10.png")
plt.close
