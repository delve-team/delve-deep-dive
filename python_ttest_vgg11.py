import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel

df = pd.read_excel("vgg11_analysis.ods", engine="odf")
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


important_thresholds = [0.9999,0.9998,0.9990000000000001,0.998,0.996,0.9940000000000001,0.99]

print ("n={}".format(data[3.0].shape[0]))

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
        if result.pvalue < p_threshold:
            start, end = "\\textbf{", "}"
        else:
            start, end = "", ""
        print(f"{start}{t:1.4g}{end} & {m:.4f}  & {sdev:.4f} & {result.statistic: .3g} & {result.pvalue:.3f} & \\\\")
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
        significant = "Yes" if result.pvalue < p_threshold else "No"
        print(f"{t:1.4g} & {m:.4f}  & {sdev:.4f} & {result.statistic: .4g} & {1-result.pvalue:.3f} & {significant} \\\\")
    except Exception as e:
        print(t, e)
        
print (",".join(list(df.thresh.unique())))
