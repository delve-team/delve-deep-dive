import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt

df = pd.read_excel("vgg19_analysis.ods", engine="odf")
#df = pd.read_csv("vgg13_analysis.csv")

# baseline = df.loc[df.thresh==3.0, ["loss", "accs", "thresh"]]

# Create dataframe
data = {}
sat_data = {}
intr_data = {}

for t in df.thresh.unique():
    #if t == 0.9990000000000001:
    #    accs = df[df.thresh==0.9990000000000001].drop_duplicates(subset=['sat_avg','loss']).accs.values.flatten()
    #else:
    accs = df.loc[df.thresh==t, ["accs"]].values.flatten()
    sat_avg = df.loc[df.thresh==t, ["sat_avg"]].values.flatten()
    intr_avg = df.loc[df.thresh==t, ["intrinsic_dimensions"]].values.flatten()
    data[t] = accs
    sat_data[t] = sat_avg
    intr_data[t] = intr_avg



#important_thresholds = [0.9999,0.9998,0.9990000000000001,0.998,0.996,0.9940000000000001,0.99]
important_thresholds = [0.9997, 0.9996,0.9995, 0.9994, 0.9993]

means = []
ranges_99 = []
errors = []

print("n: {}".format(len(data[3.0])))

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


        sat = sat_data[t].astype(np.float64)
        sat_m = np.mean(sat)
        sat_v = np.var(sat, axis, ddof=1)
        sat_sdev = np.sqrt(sat_v)
        
        intr = intr_data[t].astype(np.float64)
        intr_m = np.mean(intr)
        intr_v = np.var(intr, axis, ddof=1)
        intr_sdev = np.sqrt(intr_v)

        if result.pvalue < p_threshold:
            start, end = "\\textbf{", "}"
        else:
            start, end = "", ""
        error_dim = intr_sdev * 1.96
        print(f"{start}{t:1.4g}{end}& {m:.4f}  & {sdev:.4f} & {result.statistic: .3g} & {result.pvalue:.3f} & {sat_m:.1f} & {sat_sdev:.1f} & ${intr_m:.0f} \\pm {error_dim:.0f}$ \\\\")
    except Exception as e:
        print(t, e)
        pass

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
        sat = sat_data[t].astype(np.float64)
        sat_m = np.mean(sat)
        sat_v = np.var(sat, axis, ddof=1)
        sat_sdev = np.sqrt(sat_v)

        intr = intr_data[t].astype(np.float64)
        intr_m = np.mean(intr)
        intr_v = np.var(intr, axis, ddof=1)
        intr_sdev = np.sqrt(intr_v)
        if result.pvalue < p_threshold:
            start, end = "\\textbf{", "}"
        else:
            start, end = "", ""
        error_dim = intr_sdev * 1.96
        print(f"{t:1.4g} & {result.statistic: .3g} & {result.pvalue:.3f} & {sat_m:.1f} & {sat_sdev:.1f} & ${intr_m:.0f} \\pm {error_dim:.0f}$ \\\\")
    except Exception as e:
        print(t, e)

