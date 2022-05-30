
import re
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft, fftfreq
import pickle


tedic = pickle.load(open('DATA/tedic', 'rb'))



te_sp_ae_1_5 = [tedic[f'te_sp_v{i}'] for i in [1,2,3,4,5]]
te_ae_sp_1_5 = [tedic[f'te_v_sp{i}'] for i in [1,2,3,4,5]]

te_sp_ae = np.vstack(te_sp_ae_1_5)
te_ae_sp = np.vstack(te_ae_sp_1_5)
te = te_sp_ae - te_ae_sp

prom1 = te.mean(axis=0)
var = te.std(axis=0)



x = np.arange(0, 2499, 1)
y = prom1
plt.figure(figsize=(8, 5), dpi=200)
plt.plot(x, y, color='C0')
plt.fill_between(x, y-var, y+var, color='C0', alpha=0.3)
plt.xlim(x[0], x[-1])
plt.xlabel('Hours')
plt.ylabel('Transfer Entropy')
plt.title('Average and standard deviation')
plt.grid()
plt.legend(('Average TE sp→v - v→sp', 'Standard Deviation'),
           loc='upper right')
