
import re
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft, fftfreq
import pickle

# GRÁFICA DEL PROMEDIO DE LA TE DE LAS VENTANAS DE ALTA Y BAJA INTENSIDAD

tedic = pickle.load(open('DATA/tedic', 'rb'))

high = [1, 2, 3, 4, 5]
low = [6, 7, 8,  9]

te_sp_ae_1_5 = [tedic[f'te_sp_ae{i}'] for i in high]
te_ae_sp_1_5 = [tedic[f'te_ae_sp{i}'] for i in high]



te_sp_ae = np.vstack(te_sp_ae_1_5)
te_ae_sp = np.vstack(te_ae_sp_1_5)
te = te_sp_ae - te_ae_sp

prom1 = te.mean(axis=0)
var = te.std(axis=0)

vumbral = 4.2 * (10**(-4))


x = np.arange(0, len(prom1), 1)
y = prom1
plt.figure(figsize=(8, 5), dpi=200)
plt.plot(x, y, color='C0')
plt.fill_between(x, y-var, y+var, color='C0', alpha=0.3)
plt.hlines(vumbral, x[0], x[-1], colors= 'C1')
plt.hlines(-vumbral, x[0], x[-1], colors= 'C1')
plt.xlim(x[0], x[-1])
plt.xlabel('Hours')
plt.ylabel('Transfer Entropy')
plt.title('Average and standard deviation (High Activity)')
plt.grid()
plt.legend(('Average TE sp→AE Index - AE Index→sp', 'Standard Deviation'),
           loc='upper right')



