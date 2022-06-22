
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from Windowcreation import windowcreate
from scipy.fftpack import fft, fftfreq
import pickle
import os

''' BÚSQUEDA DE DF'''


df=pd.read_csv(os.path.join('DATA', 'dataframe.csv'))
df.info()
print(df)


''' BÚSQUEDA DE DATOS DEL DICCIONARIO VENTANAS '''

windowdic = pickle.load(open('DATA/windowdata', 'rb'))

R_max = windowdic['R_hi']
R_max1 = R_max[1]

Dst_max = windowdic['AE_hi']
Dst_max1 = Dst_max[1]

Rarray = np.array(R_max1).flatten()
Rbined = np.clip(0, 1, np.floor(2*(Rarray - Rarray.min())/(Rarray.max()-Rarray.min())))
Darray = np.array(Dst_max1).flatten()
Dbined = np.clip(0, 1, np.floor(2*(Darray - Darray.min())/(Darray.max()-Darray.min())))

df = pd.DataFrame(R_max1)
df = df.append(Dst_max1)
df = df.transpose()
print(df)
df.to_csv(os.path.join('DATA', 'R_Dstdataframe.csv'))


''' PLOT SERIES TEMPORALES'''

f1, ax = plt.subplots()
ax.plot( df["DTIME"], df['R (Sunspot No.)'])
ax.set(xlabel='time (h)', ylabel='R (Sunspot No.)',
       title='Sunspots No.(1963 - 2021)')
ax.grid()
plt.show()

f2, ax = plt.subplots()
ax.plot(df['DTIME'], df['Dst-index, nT'])
ax.set(xlabel='time (h)', ylabel='Dst-index (nT)',
       title='Dst-index (1963 - 2021)')
ax.grid()
plt.show()

f3, ax = plt.subplots()
ax.plot(df['DTIME'], df['AE-index, nT'])
ax.set(xlabel='time (h)', ylabel='AE-index (nT)',
       title='AE-index (1963 - 2021)')
ax.grid()
plt.show()

f4, ax = plt.subplots()
ax.plot(df['DTIME'], df['SW Proton Density, N/cm^3'])
ax.set(xlabel='time (h)', ylabel='SW Proton Density (N/cm^3)',
       title='Sun Wind Proton Density (1963 - 2021)')
ax.grid()
plt.show()


''' PLOT POWER SPECTRUM '''

y1 = ps_Dst1[freqfourier_Dst1>=0]
x1 = 24 * freqfourier_Dst1[freqfourier_Dst1>=0]

fig1, ax = plt.subplots(nrows=1, ncols= 2, figsize= (16, 10))

ax[0].plot(x1, y1)
ax[0].set_title('Dst Index Power Spectrum - Max. 1')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_ylim([10**-2, 10**12])
ax[0].set_ylabel('Power spectrum')
ax[0].set_xlabel('Frecuencia (1/día)')
ax[0].grid()

y2 =  ps_Dst2[freqfourier_Dst2>=0]
x2 = 24 * freqfourier_Dst2[freqfourier_Dst2>=0]

ax[1].plot( x2, y2)
ax[1].set_title('Dst Index Power Spectrum - Max. 2')
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_ylim([10**-2, 10**12])
ax[1].set_ylabel('Power spectrum')
ax[1].set_xlabel('Frecuencia (1/día)')
ax[1].grid()

fig2, ax = plt.subplots(nrows=1, ncols= 2, figsize= (16, 12))

y3 =  ps_Dstl1[freqfourier_Dstl1>=0]
x3 = 24 * freqfourier_Dstl1[freqfourier_Dstl1>=0]

ax[0].plot( x3, y3)
ax[0].set_title('Dst Index Power Spectrum - Min. 1')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_ylim([10**-2, 10**12])
ax[0].set_ylabel('Power spectrum')
ax[0].set_xlabel('Frecuencia (1/día)')
ax[0].grid()

y4 =  ps_Dstl2[freqfourier_Dstl2>=0]
x4 = 24 * freqfourier_Dstl2[freqfourier_Dstl2>=0]

ax[1].plot(x4, y4)
ax[1].set_title('Dst Index Power Spectrum - Mín. 2')
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_ylim([10**-2, 10**12])
ax[1].set_ylabel('Magnitud Power spectrum')
ax[1].set_xlabel('Frecuencia (1/día)')
ax[1].grid()

plt.tight_layout(pad=2, h_pad=1)
