# This is a sample Python script.

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from Windowcreation_and_graph import windowcreate
from scipy.fftpack import fft, fftfreq
import pickle
import os




df=pd.read_csv(os.path.join('DATA', 'dataframe.csv'))


df.info()

print(df)


# El cname es el título
cname= "R (Sunspot No.)"
df = df.iloc[:int(len(df)*1)]

def formatt(x, title, output):
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xticks(np.linspace(0, x.max(), 15))
    plt.show()
    #fig.savefig(outputfile)


x = df["DTIME"]
y = df["R (Sunspot No.)"]
w=50000 #valor para suavizar la función)
ma = np.convolve(y, np.ones(w)/w, mode='same')

dma = np.diff(ma)
madma = np.convolve(dma, np.ones(w)/w, mode='same')

# Plot de la señal suavizada, con derivada y
'''plt.figure(figsize=(8, 4))
ax = plt.gca()
ax2 = ax.twinx()
ax.plot(x, y, alpha=0.5)
ax.plot(x, ma)
ax2.plot(x[:-1], madma, color="C3")
formatt(x, cname, cname+".png")'''


maxmin=[]
madmasig=np.sign(madma)
for i in range(1, len(madma)):
    if madmasig[i]!= madmasig[i-1]:
        maxmin.append(i)
        #plt.plot(x[i], madma[i], 'ro', alpha=0.3)

print(maxmin)


#Creación de las ventanas y guardado de datos en el diccionario windowdic

sigma=10000
R_low1 = windowcreate(counter=1, maxmin=maxmin, x=x, y=y, sigma= sigma)
R_hi1 = windowcreate(0, maxmin, x, y, sigma)
R_hi2 = windowcreate(2, maxmin, x, y, sigma)
R_low2 = windowcreate(3, maxmin, x, y, sigma)
R_hi3 = windowcreate(4, maxmin, x, y, sigma)
R_low3 = windowcreate(5, maxmin, x, y, sigma)
R_hi4 = windowcreate(6, maxmin, x, y, sigma)
R_low4 = windowcreate(7, maxmin, x, y, sigma)
R_hi5 = windowcreate(8, maxmin, x, y, sigma)

Dst_low1 = windowcreate(counter=1, maxmin=maxmin, x= x, y=df['Dst-index, nT'],  sigma= sigma)
Dst_hi1 = windowcreate(0, maxmin, x, df['Dst-index, nT'], sigma)
Dst_hi2 = windowcreate(2, maxmin, x, df['Dst-index, nT'], sigma)
Dst_low2 = windowcreate(3, maxmin, x, df['Dst-index, nT'], sigma)
Dst_hi3 = windowcreate(4, maxmin, x, df['Dst-index, nT'], sigma)
Dst_low3 = windowcreate(5, maxmin, x, df['Dst-index, nT'], sigma)
Dst_hi4 = windowcreate(6, maxmin, x, df['Dst-index, nT'], sigma)
Dst_low4 = windowcreate(7, maxmin, x, df['Dst-index, nT'], sigma)
Dst_hi5 = windowcreate(8, maxmin, x, df['Dst-index, nT'], sigma)

AE_low1 = windowcreate(counter=1, maxmin=maxmin, x= x, y=df['AE-index, nT'], sigma= sigma)
AE_hi1 = windowcreate(0, maxmin, x, df['AE-index, nT'], sigma)
AE_hi2 = windowcreate(2, maxmin, x, df['AE-index, nT'], sigma)
AE_low2 = windowcreate(3, maxmin, x, df['AE-index, nT'], sigma)
AE_hi3 = windowcreate(4, maxmin, x, df['AE-index, nT'], sigma)
AE_low3 = windowcreate(5, maxmin, x, df['AE-index, nT'], sigma)
AE_hi4 = windowcreate(6, maxmin, x, df['AE-index, nT'], sigma)
AE_low4 = windowcreate(7, maxmin, x, df['AE-index, nT'], sigma)
AE_hi5 = windowcreate(8, maxmin, x, df['AE-index, nT'], sigma)


SW_low1 = windowcreate(counter=1, maxmin=maxmin, x=x, y=df['SW Proton Density, N/cm^3'], sigma= sigma)
SW_hi1 = windowcreate(0, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
SW_hi2 = windowcreate(2, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
SW_low2 = windowcreate(3, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
SW_hi3 = windowcreate(4, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
SW_low3 = windowcreate(5, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
SW_hi4 = windowcreate(6, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
SW_low4 = windowcreate(7, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
SW_hi5 = windowcreate(8, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)

v_low1 = windowcreate(counter=1, maxmin=maxmin, x=x, y=df['SW Plasma Speed, km/s'], sigma= sigma)
v_hi1 = windowcreate(0, maxmin, x, df['SW Plasma Speed, km/s'], sigma)
v_hi2 = windowcreate(2, maxmin, x, df['SW Plasma Speed, km/s'], sigma)
v_low2 = windowcreate(3, maxmin, x, df['SW Plasma Speed, km/s'], sigma)
v_hi3 = windowcreate(4, maxmin, x, df['SW Plasma Speed, km/s'], sigma)
v_low3 = windowcreate(5, maxmin, x, df['SW Plasma Speed, km/s'], sigma)
v_hi4 = windowcreate(6, maxmin, x, df['SW Plasma Speed, km/s'], sigma)
v_low4 = windowcreate(7, maxmin, x, df['SW Plasma Speed, km/s'], sigma)
v_hi5 = windowcreate(8, maxmin, x, df['SW Plasma Speed, km/s'], sigma)

b_low1 = windowcreate(counter=1, maxmin=maxmin, x=x, y=df['Scalar B, nT'], sigma= sigma)
b_hi1 = windowcreate(0, maxmin, x, df['Scalar B, nT'], sigma)
b_hi2 = windowcreate(2, maxmin, x, df['Scalar B, nT'], sigma)
b_low2 = windowcreate(3, maxmin, x, df['Scalar B, nT'], sigma)
b_hi3 = windowcreate(4, maxmin, x, df['Scalar B, nT'], sigma)
b_low3 = windowcreate(5, maxmin, x, df['Scalar B, nT'], sigma)
b_hi4 = windowcreate(6, maxmin, x, df['Scalar B, nT'], sigma)
b_low4 = windowcreate(7, maxmin, x, df['Scalar B, nT'], sigma)
b_hi5 = windowcreate(8, maxmin, x, df['Scalar B, nT'], sigma)

d_low1 = windowcreate(counter=1, maxmin=maxmin, x=x, y=df['SW Proton Density, N/cm^3'], sigma= sigma)
d_hi1 = windowcreate(0, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
d_hi2 = windowcreate(2, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
d_low2 = windowcreate(3, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
d_hi3 = windowcreate(4, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
d_low3 = windowcreate(5, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
d_hi4 = windowcreate(6, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
d_low4 = windowcreate(7, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)
d_hi5 = windowcreate(8, maxmin, x, df['SW Proton Density, N/cm^3'], sigma)


'''


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




'''
windowdic={
    'maxmin': maxmin,
    'R_hi': {
        1: R_hi1,
        2: R_hi2,
        3: R_hi3,
        4: R_hi4,
        5: R_hi5
    },
    'R_low':{
        1: R_low1,
        2: R_low2,
        3: R_low3,
        4: R_low4
    },
    'Dst_hi':{
        1: Dst_hi1,
        2: Dst_hi2,
        3: Dst_hi3,
        4: Dst_hi4,
        5: Dst_hi5
    },
    'Dst_low':{
        1: Dst_low1,
        2: Dst_low2,
        3: Dst_low3,
        4: Dst_low4,
    },
    'AE_hi':{
        1: AE_hi1,
        2: AE_hi2,
        3: AE_hi3,
        4: AE_hi4,
        5: AE_hi5
    },
    'AE_low':{
        1: AE_low1,
        2: AE_low2,
        3: AE_low3,
        4: AE_low4,
    },
     'SW_hi':{
        1: SW_hi1,
        2: SW_hi2,
        3: SW_hi3,
        4: SW_hi4,
        5: SW_hi5
    },
    'SW_low':{
        1: SW_low1,
        2: SW_low2,
        3: SW_low3,
        4: SW_low4,
    },
    'v_hi':{
        1: v_hi1,
        2: v_hi2,
        3: v_hi3,
        4: v_hi4,
        5: v_hi5
    },
    'v_low':{
        1: v_low1,
        2: v_low2,
        3: v_low3,
        4: v_low4,
    },
    'b_hi':{
        1: b_hi1,
        2: b_hi2,
        3: b_hi3,
        4: b_hi4,
        5: b_hi5
    },
    'b_low':{
        1: b_low1,
        2: b_low2,
        3: b_low3,
        4: b_low4,
    },
    'd_hi':{
        1: d_hi1,
        2: d_hi2,
        3: d_hi3,
        4: d_hi4,
        5: d_hi5
    },
    'd_low':{
        1: d_low1,
        2: d_low2,
        3: d_low3,
        4: d_low4,
    }
}

pickle.dump(windowdic, open('DATA/windowdata', 'wb'))







