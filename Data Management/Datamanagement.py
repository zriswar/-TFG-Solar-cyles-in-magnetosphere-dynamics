
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from Plotter import windowcreate
from scipy.fftpack import fft, fftfreq
import pickle
import os


df_total = pd.read_csv(os.path.join('DATA', 'dataframe.csv'))
df_total.info()

print(df_total)

df_total = df_total.iloc[:int(len(df_total)*1)]
df1 = df_total[["R (Sunspot No.)","Dst-index, nT"]]
df2 = df_total[["R (Sunspot No.)","AE-index, nT"]]


df1.to_csv(os.path.join('DATA', 'DstIndex_Sunspots.csv'))
df2.to_csv(os.path.join('DATA', 'AEIndex_Sunspots.csv'))

nsundic=pickle.load(open('DATA/windowdata', 'rb'))



