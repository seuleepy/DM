import math
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch import from_numpy

infile1 = "./prediction_01_01_wght10.csv"
infile2 = "./prediction_02_01_wght10.csv"
infile3 = "./prediction_03_01_wght10.csv"
infile4 = "./prediction_04_01_wght10.csv"
infile5 = "./prediction_05_01_wght10.csv"
infile6 = "./prediction_06_01_wght10.csv"
infile7 = "./prediction_07_01_wght10.csv"
infile8 = "./prediction_08_01_wght10.csv"
infile9 = "./prediction_09_01_wght10.csv"
infile10 = "./prediction_10_00.csv"


df1 = pd.read_csv(infile1)
df2 = pd.read_csv(infile2)
df3 = pd.read_csv(infile3)
df4 = pd.read_csv(infile4)
df5 = pd.read_csv(infile5)
df6 = pd.read_csv(infile6)
df7 = pd.read_csv(infile7)
df8 = pd.read_csv(infile8)
df9 = pd.read_csv(infile9)
df10 = pd.read_csv(infile10)

#tpr, tpr, thr = roc_curve(df['label'], df['prediction'], sample_weight = df['weight'])
#auc = roc_auc_score(df['label'], df['prediction'], sample_weight = df['weight'])

df_bkg = df1[df1.label == 0]
df_sig1 = df1[df1.label == 1]
df_sig2 = df2[df2.label == 1]
df_sig3 = df3[df3.label == 1]
df_sig4 = df4[df4.label == 1]
df_sig5 = df5[df5.label == 1]
df_sig6 = df6[df6.label == 1]
df_sig7 = df7[df7.label == 1]
df_sig8 = df8[df8.label == 1]
df_sig9 = df9[df9.label == 1]
df_sig10 = df10[df10.label == 1]

SF1 = 0.00014732148708728937 # signal01
SF2 = 0.00015605312267432672 # signal02
SF3 = 0.00015659707393823327
SF4 = 0.00015751854361664214
SF5 = 0.00015764844242593327
SF6 = 0.0001570192450684294
SF7 = 0.00015833447051250197
SF8 = 0.00015903673595023208
SF9 = 0.00016042502947453093
SF10 = 0.0001624100456540108 # signal10

lumi = 3000000
genevt = 100000 
xsec1 = 2.532e-01 #signal01
xsec2 = 3.612e-02
xsec3 = 1.223e-02
xsec4 = 5.418e-03
xsec5 = 2.504e-03
xsec6 = 1.371e-03
xsec7 = 8.212e-04
xsec8 = 5.074e-04
xsec9 = 3.292e-04
xsec10 = 2.188e-04

plt.rcParams['figure.figsize'] = (6, 6)

hbkg = plt.hist(df_bkg['prediction'], histtype = 'step',
#density = True, 
weights = df_bkg['weight']/SF1,
range = (0, 1), bins = 10000, linewidth = 3, color = 'gray', label = 'BKG')

hsig1 = plt.hist(df_sig1['prediction'], histtype = 'step',
#density = True,
weights = df_sig1['weight'] * lumi * xsec1 / genevt, 
range = (0, 1),  bins = 10000, linewidth = 3, color = 'palevioletred', label = 'SIG 100 GeV')

hsig2 = plt.hist(df_sig2['prediction'], histtype = 'step',
weights = df_sig2['weight']*lumi*xsec2/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color = 'mediumvioletred', label = 'SIG 200 GeV')

hsig3 = plt.hist(df_sig3['prediction'], histtype = 'step',
weights = df_sig3['weight']*lumi*xsec3/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color = 'darkgoldenrod', label = 'SIG 300 GeV')

hsig4 = plt.hist(df_sig4['prediction'], histtype = 'step',
weights = df_sig4['weight']*lumi*xsec4/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color = 'darkorange', label = 'SIG 400 GeV')

hsig5 = plt.hist(df_sig5['prediction'], histtype = 'step',
weights = df_sig5['weight']*lumi*xsec5/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color = 'gold', label = 'SIG 500 GeV')

hsig6 = plt.hist(df_sig6['prediction'], histtype = 'step',
weights = df_sig6['weight']*lumi*xsec6/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color = 'mediumseagreen', label = 'SIG 600 GeV')

hsig7 = plt.hist(df_sig7['prediction'], histtype = 'step',
weights = df_sig7['weight']*lumi*xsec7/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color = 'g', label = 'SIG 700 GeV')

hsig8 = plt.hist(df_sig8['prediction'], histtype = 'step',
weights = df_sig8['weight']*lumi*xsec8/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color = 'lightblue', label = 'SIG 800 GeV')

hsig9 = plt.hist(df_sig9['prediction'], histtype = 'step',
weights = df_sig9['weight']*lumi*xsec9/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color = 'steelblue', label = 'SIG 900 GeV')

hsig10 = plt.hist(df_sig10['prediction'], histtype = 'step',
weights = df_sig10['weight']*lumi*xsec10/genevt,
range = (0, 1), bins = 10000, linewidth = 3, color ='slateblue', label = 'SIG 1 TeV')

'''
plt.xlabel('DNN score', fontsize = 17)
plt.ylabel('Events', fontsize = 17)
plt.legend(loc = "best", fontsize = 6)
plt.grid()
plt.yscale('log')
plt.savefig("DNN_score_all_wght10.png")
plt.close()

plt.plot(fpr, tpr, '.', linewidth = 2, label = '%s %.3f' %("auc", auc))
plt.xlim(0, 1.000)
plt.ylim(0, 1.000)
plt.xlabel('False Positive Rate', fontsize = 17)
plt.ylabel('True Positive Rate', fontsize = 17)
plt.legend(fontsize = 17)
plt.savefig("ROC_01_01_wght10.png")
plt.close()
'''
N_bkg = hbkg[0]
N_sig = hsig1[0]


arr_significance = []
print("cut, sig_intergral, bkg_integral, significance\n---------------------------------------------\n")
for cut in range(5000, len(N_bkg), 1):
	sig_integral = sum(N_sig[cut:])
	bkg_integral = sum(N_bkg[cut:])
	
	if sig_integral + bkg_integral == 0:
		significance = 0
	else:
		significance = sig_integral / math.sqrt(sig_integral + bkg_integral)
	arr_significance.append(significance)
	
	print("[", cut*0.0001, ":]", sig_integral, bkg_integral, significance)

dnn_idx = arr_significance.index(max(arr_significance)) + 5000
print("DNN score : [", dnn_idx*0.0001, ":]")
print("significance :", max(arr_significance))

N_sig_ls = ["hsig" + str(i) for i in range(1, 11)]


for i in N_sig_ls:
	print("sig_file sig_integral, significance\n-------------------------------------\n")
	sig_integral = sum(eval(i)[0][5061:])
	bkg_integral = 489898.95243616047

	if sig_integral + bkg_integral == 0: significance = 0
	else: significance = sig_integral / math.sqrt(sig_integral + bkg_integral)

	print(i, sig_integral, significance,"\n")

'''
plt.rcParams["legend.loc"] = "lower left"
plt.plot(list(i * 0.0001 for i in range(0, 10000)), arr_significance, '-o', color = 'dimgray')
plt.xlabel('DNN score', fontsize = 25)
plt.ylabel('Significance', fontsize = 25)
plt.grid(which = 'major' , linestyle = '-')
plt.minorticks_on()
plt.savefig('sig_01_01_wght10')
'''
