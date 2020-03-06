import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt

df = pd.read_excel("vgg13_analysis.ods", engine="odf")
#df = pd.read_csv("vgg13_analysis.csv")

# baseline = df.loc[df.thresh==3.0, ["loss", "accs", "thresh"]]

# Create dataframe
data = {}

for t in df.thresh.unique():
    #if t == 0.9990000000000001:
    #    accs = df[df.thresh==0.9990000000000001].drop_duplicates(subset=['sat_avg','loss']).accs.values.flatten()
    #else:
    accs = df.loc[df.thresh==t, ["accs"]].values.flatten()
    data[t] = accs


important_thresholds = [0.9999,0.9998,0.999,0.998,0.996,0.994,0.99]

means = []
ranges_99 = []
errors = []
print(df.thresh.unique())
print(f"Thresh & \mu & \sigma & t-stat & p-value \\\\")
for t in df.thresh.unique():
    try:
        a, b = data[3.0], data[t]
        axis = 0
        p_threshold = 1-.99
        result = ttest_rel(a, b) # slice because final value is duplicate in one array
        d = (a - b).astype(np.float64)
        m = np.mean(d, axis)
        v = np.var(d, axis, ddof=1)
        sdev = np.sqrt(v)
        means.append(1-m)
        errors.append(2.58 * sdev / np.sqrt(26))
        significant = "Yes" if result.pvalue < p_threshold else "No"
        if result.pvalue < p_threshold:
            start, end = "\\textbf{", "}"
        else:
            start, end = "", ""
        print(f"{start}{t:1.4g}{end} & {m:.4f}  & {sdev:.4f} & {result.statistic: .3g} & {result.pvalue:.9f}  \\\\")
    except Exception as e:
        print(t, e)

print("INLINE GRAPH\n\n")
for t in df.thresh.unique():
    if t not in important_thresholds:
        continue
    try:
        a, b = data[3.0], data[t]
        axis = 0
        p_threshold = 1-.99
        result = ttest_rel(a, b) # slice because final value is duplicate in one array
        d = (a - b).astype(np.float64)
        m = np.mean(d, axis)
        v = np.var(d, axis, ddof=1)
        sdev = np.sqrt(v)
        if result.pvalue < p_threshold:
            start, end = "\\textbf{", "}"
        else:
            start, end = "", ""
        print(f"{start}{t:1.4g}{end} & {m:.4f}  & {sdev:.4f} & {result.statistic: .3g} & {result.pvalue:.3f} \\\\")
    except Exception as e:
        print(t, e)
        
#print (",".join(list(df.thresh.unique())))

#means = np.array(means)
#errors = np.array(errors)
#ax = df.thresh.unique()
#ax[0] = 1.0
#print(df.thresh.unique()[1:])
#plt.plot(ax[:15], means[:15])
#plt.errorbar(ax[:15], means[:15], yerr=errors[:15])
##plt.xscale('log')
#plt.show()
