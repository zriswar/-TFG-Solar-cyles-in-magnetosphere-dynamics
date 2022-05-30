# This is a sample Python script.

import re
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt




with open('DATA/omni2_Q5hn2Xq4cF.lst', 'r') as f:
    file = f.readlines()

def format(r):
    return re.sub('\s+', ' ', r).split(' ')[:-1]

a = np.array(list(map(format, file)), dtype=float)
df=pd.DataFrame(a, columns=["YEAR", "DOY", "Hour", "Scalar B, nT", "BZ, nT (GSE)", "SW Proton Density, N/cm^3", "SW Plasma Speed, km/s", "R (Sunspot No.)", "Dst-index, nT", "f10.7_index", "AE-index, nT"])


time=8760*df["YEAR"]+24*df["DOY"]+df["Hour"]
df["DTIME"]= time - time.iloc[0]


errordata=[999.9, 999.9, 999.9, 9999., 999, 99999, 999.9, 9999]
column =['Scalar B, nT', 'BZ, nT (GSE)', 'SW Proton Density, N/cm^3', 'SW Plasma Speed, km/s', 'R (Sunspot No.)', 'Dst-index, nT', 'f10.7_index', 'AE-index, nT']
changes = []

# Datafilling with forwardfill
'''for c, e in zip(column, errordata):
    changes.append(np.sum(df[c]==e))
    df[c] = df[c].replace(e, method = 'ffill')'''

# Datafilling with prom.
for c, e in zip(column, errordata):
    prev = []
    pos = []
    tot = 0
    alfa = 100
    changes.append(np.sum(df[c]==e))

    for i in range(0, len(df[c])):
        if df[c][i] == e and len(prev) == 0:
            prev = df[c][max(i-alfa, 0) : i]
            tot=1
        elif df[c][i] == e and len(prev) > 0:
            tot = tot + 1
        elif df[c][i] != e and len(prev) > 0:
            pos = df[c][i : max(i + alfa, len(c))]
            pos = pos[pos!=e]

            media = np.concatenate((prev.values, pos.values)).mean()
            df[c][i-tot : i] = media
            tot = 0
            prev = []
            pos = []


#df.to_csv(os.path.join('DATA', 'dataframe.csv'))








