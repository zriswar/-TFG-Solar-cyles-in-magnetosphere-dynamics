# This is a sample Python script.

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



# FOURIER

def fourier(signal, ):
    signal = signal - np.mean(signal)
    transfourier= np.fft.fft(signal).flatten()
    n=transfourier.size
    freqtfourier= np.fft.fftfreq(n, )
    return transfourier, freqtfourier

#def freqfourier (signal, ):
    n=signal.size
    freqtfourier= np.fft.fftfreq(n, )
    return freqtfourier

# Transformadas de Fourier Sunspots

fourier_R1, freqfourier_R1 = fourier(R_hi1)
mod_R1 = np.abs(fourier_R1)
ps_R1 = mod_R1**2 #Power Spectrum

fourier_R2, freqfourier_R2 = fourier(R_hi2)
mod_R2 = np.abs(fourier_R2)
ps_R2 = mod_R2**2

fourier_R3, freqfourier_R3 = fourier(R_hi3)
mod_R3 = np.abs(fourier_R3)
ps_R3 = mod_R3**2

fourier_R4, freqfourier_R4 = fourier(R_hi4)
mod_R4 = np.abs(fourier_R4)
ps_R4 = mod_R4**2

fourier_R5, freqfourier_R5 = fourier(R_hi5)
mod_R5 = np.abs(fourier_R5)
ps_R5 = mod_R5**2


fourier_Rl1, freqfourier_Rl1 = fourier (R_low1)
mod_Rl1 = np.abs(fourier_R1)
ps_Rl1 = mod_Rl1**2 #Power Spectrum

fourier_Rl2, freqfourier_Rl2 = fourier (R_low2)
mod_Rl2 = np.abs(fourier_R2)
ps_Rl2 = mod_Rl2**2

fourier_Rl3, freqfourier_Rl3 = fourier (R_low3)
mod_Rl3 = np.abs(fourier_R3)
ps_Rl3 = mod_Rl3**2

fourier_Rl4, freqfourier_Rl4 = fourier (R_low4)
mod_Rl4 = np.abs(fourier_R4)
ps_Rl4 = mod_Rl4**2


# Transformadas de Fourier Dst

fourier_Dst1, freqfourier_Dst1 = fourier(Dst_hi1)
mod_Dst1 = np.abs(fourier_Dst1)
ps_Dst1 = mod_Dst1**2 #Power Spectrum

fourier_Dst2, freqfourier_Dst2 = fourier(Dst_hi2)
mod_Dst2 = np.abs(fourier_Dst2)
ps_Dst2 = mod_Dst2**2

fourier_Dst3, freqfourier_Dst3 = fourier(Dst_hi3)
mod_Dst3 = np.abs(fourier_Dst3)
ps_Dst3 = mod_Dst3**2

fourier_Dst4, freqfourier_Dst4 = fourier(Dst_hi4)
mod_Dst4 = np.abs(fourier_Dst4)
ps_Dst4 = mod_Dst4**2

fourier_Dst5, freqfourier_Dst5 = fourier(Dst_hi5)
mod_Dst5 = np.abs(fourier_Dst5)
ps_Dst5 = mod_Dst5**2


fourier_Dstl1, freqfourier_Dstl1 = fourier (Dst_low1)
mod_Dstl1 = np.abs(fourier_Dst1)
ps_Dstl1 = mod_Dstl1**2 #Power Spectrum

fourier_Dstl2, freqfourier_Dstl2 = fourier (Dst_low2)
mod_Dstl2 = np.abs(fourier_Dst2)
ps_Dstl2 = mod_Dstl2**2

fourier_Dstl3, freqfourier_Dstl3 = fourier (Dst_low3)
mod_Dstl3 = np.abs(fourier_Dst3)
ps_Dstl3 = mod_Dstl3**2

fourier_Dstl4, freqfourier_Dstl4 = fourier (Dst_low4)
mod_Dstl4 = np.abs(fourier_Dst4)
ps_Dstl4 = mod_Dstl4**2

# Transformadas de Fourier AE

fourier_AE1, freqfourier_AE1 = fourier(AE_hi1)
mod_AE1 = np.abs(fourier_AE1)
ps_AE1 = mod_AE1**2 #Power Spectrum

fourier_AE2, freqfourier_AE2 = fourier(AE_hi2)
mod_AE2 = np.abs(fourier_AE2)
ps_AE2 = mod_AE2**2

fourier_AE3, freqfourier_AE3 = fourier(AE_hi3)
mod_AE3 = np.abs(fourier_AE3)
ps_AE3 = mod_AE3**2

fourier_AE4, freqfourier_AE4 = fourier(AE_hi4)
mod_AE4 = np.abs(fourier_AE4)
ps_AE4 = mod_AE4**2

fourier_AE5, freqfourier_AE5 = fourier(AE_hi5)
mod_AE5 = np.abs(fourier_AE5)
ps_AE5 = mod_AE5**2


fourier_AEl1, freqfourier_AEl1 = fourier (AE_low1)
mod_AEl1 = np.abs(fourier_AE1)
ps_AEl1 = mod_AEl1**2 #Power Spectrum

fourier_AEl2, freqfourier_AEl2 = fourier (AE_low2)
mod_AEl2 = np.abs(fourier_AE2)
ps_AEl2 = mod_AEl2**2

fourier_AEl3, freqfourier_AEl3 = fourier (AE_low3)
mod_AEl3 = np.abs(fourier_AE3)
ps_AEl3 = mod_AEl3**2

fourier_AEl4, freqfourier_AEl4 = fourier (AE_low4)
mod_AEl4 = np.abs(fourier_AE4)
ps_AEl4 = mod_AEl4**2

# Transformada de Fourier SW

fourier_SW1, freqfourier_SW1 = fourier(SW_hi1)
mod_SW1 = np.abs(fourier_SW1)
ps_SW1 = mod_SW1**2 #Power Spectrum

fourier_SW2, freqfourier_SW2 = fourier(SW_hi2)
mod_SW2 = np.abs(fourier_SW2)
ps_SW2 = mod_SW2**2

fourier_SW3, freqfourier_SW3 = fourier(SW_hi3)
mod_SW3 = np.abs(fourier_SW3)
ps_SW3 = mod_SW3**2

fourier_SW4, freqfourier_SW4 = fourier(SW_hi4)
mod_SW4 = np.abs(fourier_SW4)
ps_SW4 = mod_SW4**2

fourier_SW5, freqfourier_SW5 = fourier(SW_hi5)
mod_SW5 = np.abs(fourier_SW5)
ps_SW5 = mod_SW5**2


fourier_SWl1, freqfourier_SWl1 = fourier (SW_low1)
mod_SWl1 = np.abs(fourier_SW1)
ps_SWl1 = mod_SWl1**2 #Power Spectrum

fourier_SWl2, freqfourier_SWl2 = fourier (SW_low2)
mod_SWl2 = np.abs(fourier_SW2)
ps_SWl2 = mod_SWl2**2

fourier_SWl3, freqfourier_SWl3 = fourier (SW_low3)
mod_SWl3 = np.abs(fourier_SW3)
ps_SWl3 = mod_SWl3**2

fourier_SWl4, freqfourier_SWl4 = fourier (SW_low4)
mod_SWl4 = np.abs(fourier_SW4)
ps_SWl4 = mod_SWl4**2



# Grafica de POWER SPECTRUM

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


#pickle.dump(fig, open('../Gráficos/R Powerspectrum (con prom)', 'wb'))
# plt.savefig('../Gráficos/R Powerspectrum (con prom).png')


# Cross Spectrum Dst - Sunspots Max1
complexcs_R1_Dst1 = fourier_Dst1 * np.conjugate(fourier_R1)
mod_cs_R1_Dst1 = np.abs(complexcs_R1_Dst1)
norm_cs_R1_Dst1 = mod_cs_R1_Dst1 / np.sqrt(ps_Dst1 * ps_R1)
freq_cs_Dst1 = np.fft.fftfreq(norm_cs_R1_Dst1.size, )

phase_R1_Dst1 = np.arctan(np.imag(complexcs_R1_Dst1) / np.real(complexcs_R1_Dst1))
phasefreq_R1_Dst1 = np.fft.fftfreq(phase_R1_Dst1.size, )


# Cross Spectrum Dst - Sunspots Max2

complexcs_R2_Dst2 = fourier_Dst2 * np.conjugate(fourier_R2)
mod_cs_R2_Dst2 = np.abs(complexcs_R2_Dst2)
norm_cs_R2_Dst2 = mod_cs_R2_Dst2 / np.sqrt(ps_Dst2 * ps_R2)

freq_cs_Dst2 = np.fft.fftfreq(norm_cs_R2_Dst2.size,)

angle_R2_Dst2 = np.angle(complexcs_R2_Dst2)
phase_R2_Dst2 = np.arctan(np.imag(complexcs_R2_Dst2) / np.real(complexcs_R2_Dst2))
phasefreq_R2_Dst2 = np.fft.fftfreq(phase_R2_Dst2.size, )

# Cross Spectrum Dst - Sunspots Min. 1
complexcs_Rl1_Dstl1 = fourier_Dstl1 * np.conjugate(fourier_Rl1)
mod_cs_Rl1_Dstl1 = np.abs(complexcs_Rl1_Dstl1)
norm_cs_Rl1_Dstl1 = mod_cs_Rl1_Dstl1 / np.sqrt(ps_Dstl1 * ps_Rl1)
freq_cs_Dstl1 = np.fft.fftfreq(norm_cs_Rl1_Dstl1.size, )

phase_Rl1_Dstl1 = np.arctan(np.imag(complexcs_Rl1_Dstl1) / np.real(complexcs_Rl1_Dstl1))
phasefreq_Rl1_Dstl1 = np.fft.fftfreq(phase_Rl1_Dstl1.size, )


# Cross Spectrum Dst - Sunspots Max2

complexcs_Rl2_Dstl2 = fourier_Dstl2 * np.conjugate(fourier_Rl2)
mod_cs_Rl2_Dstl2 = np.abs(complexcs_Rl2_Dstl2)
norm_cs_Rl2_Dstl2 = mod_cs_Rl2_Dstl2 / np.sqrt(ps_Dstl2 * ps_Rl2)

freq_cs_Dstl2 = np.fft.fftfreq(norm_cs_Rl2_Dstl2.size,)

angle_Rl2_Dstl2 = np.angle(complexcs_Rl2_Dstl2)
phase_Rl2_Dstl2 = np.arctan(np.imag(complexcs_Rl2_Dstl2) / np.real(complexcs_Rl2_Dstl2))
phasefreq_Rl2_Dstl2 = np.fft.fftfreq(phase_Rl2_Dstl2.size, )

# Cross Spectrum AE - Sunspots Max1
complexcs_R1_AE1 = fourier_AE1* np.conjugate(fourier_R1)
mod_cs_R1_AE1 = np.abs(complexcs_R1_AE1 )
norm_cs_R1_AE1 = mod_cs_R1_AE1 / np.sqrt(ps_AE1 * ps_R1)

freq_cs_AE1 = np.fft.fftfreq(norm_cs_R1_AE1.size, )
angle_R1_AE1 = np.angle(complexcs_R1_AE1)
phase_R1_AE1 = np.arctan(np.imag(complexcs_R1_AE1) / np.real(complexcs_R1_AE1))
phasefreq_R1_AE1 = np.fft.fftfreq(phase_R1_AE1.size, )



# Cross Spectrum AE - Sunspots Max2

complexcs_R2_AE2 = fourier_AE2* np.conjugate(fourier_R2)
mod_cs_R2_AE2 = np.abs(complexcs_R2_AE2 )
norm_cs_R2_AE2 = mod_cs_R2_AE2 / np.sqrt(ps_AE2 * ps_R2)
freq_cs_AE2 = np.fft.fftfreq(complexcs_R2_AE2.size, )

angle_R2_AE2 = np.angle(complexcs_R2_AE2)
phase_R2_AE2 = np.arctan(np.imag(complexcs_R2_AE2) / np.real(complexcs_R2_AE2))
phasefreq_R2_AE2 = np.fft.fftfreq(phase_R2_AE2.size, )


# Cross Spectrum AE - Sunspots Min 3
complexcs_Rl3_AEl3 = fourier_AEl3* np.conjugate(fourier_Rl3)
mod_cs_Rl3_AEl3 = np.abs(complexcs_Rl3_AEl3 )
norm_cs_Rl3_AEl3 = mod_cs_Rl3_AEl3 / np.sqrt(ps_AEl3 * ps_Rl3)

freq_cs_AEl3 = np.fft.fftfreq(norm_cs_Rl3_AEl3.size, )
angle_Rl3_AEl3 = np.angle(complexcs_Rl3_AEl3)
phase_Rl3_AEl3 = np.arctan(np.imag(complexcs_Rl3_AEl3) / np.real(complexcs_Rl3_AEl3))
phasefreq_Rl3_AEl3 = np.fft.fftfreq(phase_Rl3_AEl3.size, )



# Cross Spectrum AE - Sunspots Min 4

complexcs_Rl4_AEl4 = fourier_AEl4* np.conjugate(fourier_Rl4)
mod_cs_Rl4_AEl4 = np.abs(complexcs_Rl4_AEl4 )
norm_cs_Rl4_AEl4 = mod_cs_Rl4_AEl4 / np.sqrt(ps_AEl4 * ps_Rl4)
freq_cs_AEl4 = np.fft.fftfreq(complexcs_Rl4_AEl4.size, )

angle_Rl4_AEl4 = np.angle(complexcs_Rl4_AEl4)
phase_Rl4_AEl4 = np.arctan(np.imag(complexcs_Rl4_AEl4) / np.real(complexcs_Rl4_AEl4))
phasefreq_Rl4_AEl4 = np.fft.fftfreq(phase_Rl4_AEl4.size, )

fig3, axes = plt.subplots(nrows=1, ncols= 2, figsize= (16, 10))

y1 = mod_cs_Rl1_Dstl1 [freq_cs_Dstl1>0]
x1 = 24 * freq_cs_Dstl1 [freq_cs_Dstl1>0]

axes[0].plot(x1, y1)
axes[0].set_title('Nº Sunspots - Dst Cross Spectrum (Min. 1)')
axes[0].set_ylabel('Cross spectrum')
axes[0].set_yscale('log')
axes[0].set_xlabel('Frecuencia (1/día)')
axes[0].set_xscale('log')
axes[0].grid()

y2 = mod_cs_Rl2_Dstl2 [freq_cs_Dstl2>0]
x2 = 24 * freq_cs_Dstl2 [freq_cs_Dstl2>0]

axes[1].plot(x2,y2)
axes[1].set_title('Nº Sunspots - Dst Cross Spectrum (Mín. 2)')
axes[1].set_ylabel('Magnitud Cross spectrum')
axes[1].set_yscale('log')
axes[1].set_xlabel('Frecuencia (1/h)')
axes[1].set_xscale('log')
axes[1].grid()

fig5, axes = plt.subplots(nrows=1, ncols= 2, figsize= (16, 10))

y1 = mod_cs_Rl3_AEl3 [freq_cs_AEl3>0]
x1 = 24 * freq_cs_AEl3 [freq_cs_AEl3>0]

axes[0].plot(x1, y1)
axes[0].set_title('Nº Sunspots - AE Cross Spectrum (Min. 3)')
axes[0].set_ylabel('Cross spectrum')
axes[0].set_yscale('log')
axes[0].set_xlabel('Frecuencia (1/día)')
axes[0].set_xscale('log')
axes[0].grid()

y2 = mod_cs_Rl4_AEl4 [freq_cs_AEl4>0]
x2 = 24 * freq_cs_AEl4 [freq_cs_AEl4>0]

axes[1].plot(x2,y2)
axes[1].set_title('Nº Sunspots - AE Cross Spectrum (Mín. 4)')
axes[1].set_ylabel('Magnitud Cross spectrum')
axes[1].set_yscale('log')
axes[1].set_xlabel('Frecuencia (1/día)')
axes[1].set_xscale('log')
axes[1].grid()



fig4, axes = plt.subplots(nrows=1, ncols= 2, figsize= (16, 10))

y1_2 = phase_R1_Dst1
x1_2 = phasefreq_R1_Dst1

axes[0].plot(x1_2,y1_2, '.')
axes[0].set_title('Nº Sunspots - Dst Cross Spectrum Phase (Máx. 1)')
axes[0].set_ylabel('Cross spectrum phase (radians)')
axes[0].set_xlabel('Frecuencia (1/h)')
axes[0].set_xscale('log')
axes[0].grid()

y2_2 = phase_R2_Dst2[phasefreq_R2_Dst2>0]
x2_2 = phasefreq_R2_Dst2 [phasefreq_R2_Dst2>0]

axes[1].plot(x2_2, y2_2,'.')
axes[1].set_title('Nº Sunspots - Dst Cross Spectrum Phase (Máx. 2)')
axes[1].set_ylabel('Cross spectrum phase (radians)')
axes[1].set_xlabel('Frecuencia (1/h)')
axes[1].set_xscale('log')
axes[1].grid()

#formatt(x, cname, cname+".png")
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
    }
}

pickle.dump(windowdic, open('DATA/windowdata', 'wb'))

'''
fourierdic={
    'fourier_R':{
        'hi1': fourier_R1,
        'hi2': fourier_R2,
        'hi3': fourier_R3,
        'hi4': fourier_R4,
        'hi5': fourier_R5,
        'low1': fourier_Rl1,
        'low2': fourier_Rl2,
        'low3': fourier_Rl3,
        'low4': fourier_Rl4,

    },

    'ffreq_R': {
        'hi1': freqfourier_R1,
        'hi2': freqfourier_R2,
        'hi3': freqfourier_R3,
        'hi4': freqfourier_R4,
        'hi5': freqfourier_R5,
        'low1': freqfourier_Rl1,
        'low2': freqfourier_Rl2,
        'low3': freqfourier_Rl3,
        'low4': freqfourier_Rl4,
    },

    'mod_R': {
        'hi1': mod_R1,
        'hi2': mod_R2,
        'hi3': mod_R3,
        'hi4': mod_R4,
        'hi5': mod_R5,
        'low1': mod_Rl1,
        'low2': mod_Rl2,
        'low3': mod_Rl3,
        'low4': mod_Rl4,

    },

    'fourier_Dst':{
        'hi1': fourier_Dst1,
        'hi2': fourier_Dst2,
        'hi3': fourier_Dst3,
        'hi4': fourier_Dst4,
        'hi5': fourier_Dst5,
        'low1': fourier_Dstl1,
        'low2': fourier_Dstl2,
        'low3': fourier_Dstl3,
        'low4': fourier_Dstl4,

    },

    'ffreq_Dst': {
        'hi1': freqfourier_Dst1,
        'hi2': freqfourier_Dst2,
        'hi3': freqfourier_Dst3,
        'hi4': freqfourier_Dst4,
        'hi5': freqfourier_Dst5,
        'low1': freqfourier_Dstl1,
        'low2': freqfourier_Dstl2,
        'low3': freqfourier_Dstl3,
        'low4': freqfourier_Dstl4,
    },

    'mod_Dst': {
        'hi1': mod_Dst1,
        'hi2': mod_Dst2,
        'hi3': mod_Dst3,
        'hi4': mod_Dst4,
        'hi5': mod_Dst5,
        'low1': mod_Dstl1,
        'low2': mod_Dstl2,
        'low3': mod_Dstl3,
        'low4': mod_Dstl4,

    },

    'fourier_AE':{
        'hi1': fourier_AE1,
        'hi2': fourier_AE2,
        'hi3': fourier_AE3,
        'hi4': fourier_AE4,
        'hi5': fourier_AE5,
        'low1': fourier_AEl1,
        'low2': fourier_AEl2,
        'low3': fourier_AEl3,
        'low4': fourier_AEl4,

    },

    'ffreq_AE': {
        'hi1': freqfourier_AE1,
        'hi2': freqfourier_AE2,
        'hi3': freqfourier_AE3,
        'hi4': freqfourier_AE4,
        'hi5': freqfourier_AE5,
        'low1': freqfourier_AEl1,
        'low2': freqfourier_AEl2,
        'low3': freqfourier_AEl3,
        'low4': freqfourier_AEl4,
    },

    'mod_AE': {
        'hi1': mod_AE1,
        'hi2': mod_AE2,
        'hi3': mod_AE3,
        'hi4': mod_AE4,
        'hi5': mod_AE5,
        'low1': mod_AEl1,
        'low2': mod_AEl2,
        'low3': mod_AEl3,
        'low4': mod_AEl4,

    },

    'fourier_SW':{
        'hi1': fourier_SW1,
        'hi2': fourier_SW2,
        'hi3': fourier_SW3,
        'hi4': fourier_SW4,
        'hi5': fourier_SW5,
        'low1': fourier_SWl1,
        'low2': fourier_SWl2,
        'low3': fourier_SWl3,
        'low4': fourier_SWl4,

    },

    'ffreq_SW': {
        'hi1': freqfourier_SW1,
        'hi2': freqfourier_SW2,
        'hi3': freqfourier_SW3,
        'hi4': freqfourier_SW4,
        'hi5': freqfourier_SW5,
        'low1': freqfourier_SWl1,
        'low2': freqfourier_SWl2,
        'low3': freqfourier_SWl3,
        'low4': freqfourier_SWl4,
    },

    'mod_SW': {
        'hi1': mod_SW1,
        'hi2': mod_SW2,
        'hi3': mod_SW3,
        'hi4': mod_SW4,
        'hi5': mod_SW5,
        'low1': mod_SWl1,
        'low2': mod_SWl2,
        'low3': mod_SWl3,
        'low4': mod_SWl4,

    },
}

pickle.dump(fourierdic, open('DATA/fourierdata', 'wb'))
# nsundic=pickle.load(open('DATA/windowdata', 'rb'))'''





