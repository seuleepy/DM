import math
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch import from_numpy

infile = "./prediction_01_01_wght10.csv"

df = pd.read_csv(infile)

fpr, tpr, thr = roc_curve(df['label'], df['prediction'], sample_weight = df['weight'])
auc = roc_auc_score(df['label'], df['prediction'], sample_weight = df['weight'])

df_bkg = df[df.label == 0]
df_sig = df[df.label == 1]

SF = 0.00014732148708728937 # signal01
#SF = 0.00015764844242593327 # signal05 
#SF = 0.0001624100456540108 # signal10
lumi = 3000000
genevt = 100000 
xsec = 2.532e-01 #signal01
#xsec = 2.504e-03 signal05
#xsec = 2.188e-04 #signal10

plt.rcParams['figure.figsize'] = (6, 6)

hbkg = plt.hist(df_bkg['prediction'], histtype = 'step',
#density = True, 
weights = df_bkg['weight']/SF,
range = (0, 1), bins = 50, linewidth = 3, color = 'crimson', label = 'BKG')
hsig = plt.hist(df_sig['prediction'], histtype = 'step',
#density = True,
weights = df_sig['weight'] * lumi * xsec / genevt, 
range = (0, 1),  bins = 50, linewidth = 3, color = 'royalblue', label = 'SIG')

plt.xlabel('DNN score', fontsize = 17)
plt.ylabel('Events', fontsize = 17)
plt.legend(fontsize = 15)
plt.grid()
plt.yscale('log')
plt.savefig("DNN_score_01_01_wght10.png")
plt.close()

plt.plot(fpr, tpr, '.', linewidth = 2, label = '%s %.3f' %("auc", auc))
plt.xlim(0, 1.000)
plt.ylim(0, 1.000)
plt.xlabel('False Positive Rate', fontsize = 17)
plt.ylabel('True Positive Rate', fontsize = 17)
plt.legend(fontsize = 17)
plt.savefig("ROC_01_01_wght10.png")
plt.close()

N_bkg = hbkg[0]
N_sig = hsig[0]

print(len(N_bkg))

arr_significance = []
print("cut, sig_intergral, bkg_integral, significance\n---------------------------------------------\n")
for cut in range(0, len(N_bkg), 1):
	sig_integral = sum(N_sig[cut:])
	bkg_integral = sum(N_bkg[cut:])
	
	if sig_integral + bkg_integral == 0:
		significance = 0
	else:
		significance = sig_integral / math.sqrt(sig_integral + bkg_integral)
	arr_significance.append(significance)
	
	print("[", cut*0.02, ":]", sig_integral, bkg_integral, significance)

dnn_idx = arr_significance.index(max(arr_significance))
print("DNN score : [", dnn_idx*0.02, ":]")
print("significance :", max(arr_significance))


plt.rcParams["legend.loc"] = "lower left"
plt.plot(list(i * 0.02 for i in range(0, 50)), arr_significance, '-o', color = 'dimgray')
plt.xlabel('DNN score', fontsize = 25)
plt.ylabel('Significance', fontsize = 25)
plt.grid(which = 'major' , linestyle = '-')
plt.minorticks_on()
plt.savefig('sig_01_01_wght10')
