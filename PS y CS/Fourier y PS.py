import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.fftpack import fft, fftfreq
import pickle
import os


fourierdic = pickle.load(open('DATA/fourierdata', 'rb'))
'''windowdic= pickle.load(open('DATA/windowdata', 'rb'))
windowdic = pickle.dump(windowdic, open('DATA/windowdata', 'rb'))

R_hi = windowdic['R_hi']
R_hi1 = R_hi[1]'''



# FOURIER

def fourier(signal, ):
    signal = signal - np.mean(signal)
    transfourier= np.fft.fft(signal).flatten()
    n=transfourier.size
    freqtfourier= np.fft.fftfreq(n, )
    return transfourier, freqtfourier



# Función Graph PS

def psgraph (seriesname, fourier_Dst1, freqfourier_Dst1, fourier_Dst2, freqfourier_Dst2, fourier_Dstl1, freqfourier_Dstl1, fourier_Dstl2, freqfourier_Dstl2):
    '''
    Se van a calcular y graficar paralelamente el PS para 4 señales o ventanas
    :param seriesname: 'nombre de la serie temporal para el titulo'
    :param fourier_Dst1: transformada de fourier de la señal 1
    :param freqfourier_Dst1: frecuencia asociada a la transformada de fourier de la señal 1
    :param fourier_Dst2: transformada de fourier de la señal 2
    :param freqfourier_Dst2: frecuencia asociada a la transformada de fourier de la señal 2
    :param fourier_Dstl1: transformada de fourier de la señal 3
    :param freqfourier_Dstl1: frecuencia asociada a la transformada de fourier de la señal 3
    :param fourier_Dstl2: transformada de fourier de la señal 4
    :param freqfourier_Dstl2: frecuencia asociada a la transformada de fourier de la señal 4
    :return:
    '''

    mod_Dst1 = np.abs(fourier_Dst1)
    ps_Dst1 = mod_Dst1**2

    mod_Dst2 = np.abs(fourier_Dst2)
    ps_Dst2 = mod_Dst2**2

    mod_Dstl1 = np.abs(fourier_Dstl1)
    ps_Dstl1 = mod_Dstl1**2

    mod_Dstl2 = np.abs(fourier_Dstl2)
    ps_Dstl2 = mod_Dstl2**2

    y1 = ps_Dst1[freqfourier_Dst1>=0]
    x1 = 24 * freqfourier_Dst1[freqfourier_Dst1>=0]

    fig1, ax = plt.subplots(nrows=1, ncols= 2, figsize= (16, 10))

    ax[0].plot(x1, y1)
    ax[0].set_title(seriesname+' Power Spectrum - Max. 1')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_ylim([10**-2, 10**12])
    ax[0].set_ylabel('Power spectrum')
    ax[0].set_xlabel('Frecuencia (1/día)')
    ax[0].grid()

    y2 = ps_Dst2[freqfourier_Dst2>=0]
    x2 = 24 * freqfourier_Dst2[freqfourier_Dst2>=0]

    ax[1].plot(x2, y2)
    ax[1].set_title(seriesname+'  Power Spectrum - Max. 2')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_ylim([10**-2, 10**12])
    ax[1].set_ylabel('Power spectrum')
    ax[1].set_xlabel('Frecuencia (1/día)')
    ax[1].grid()

    fig2, ax = plt.subplots(nrows=1, ncols= 2, figsize= (16, 12))

    y3 = ps_Dstl1[freqfourier_Dstl1>=0]
    x3 = 24 * freqfourier_Dstl1[freqfourier_Dstl1>=0]

    ax[0].plot(x3, y3)
    ax[0].set_title(seriesname+'  Power Spectrum - Min. 1')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_ylim([10**-2, 10**12])
    ax[0].set_ylabel('Power spectrum')
    ax[0].set_xlabel('Frecuencia (1/día)')
    ax[0].grid()

    y4 = ps_Dstl2[freqfourier_Dstl2>=0]
    x4 = 24 * freqfourier_Dstl2[freqfourier_Dstl2>=0]

    ax[1].plot(x4, y4)
    ax[1].set_title(seriesname+'  Power Spectrum - Mín. 2')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_ylim([10**-2, 10**12])
    ax[1].set_ylabel('Magnitud Power spectrum')
    ax[1].set_xlabel('Frecuencia (1/día)')
    ax[1].grid()

    plt.tight_layout(pad=2, h_pad=1)


#bCálculo cross spectrum teniendo las transformadas de fourier de la señal 1 y la 2

def crossspectrum(fourier_Dst1, fourier_R1):
    '''
    :param fourier_Dst1: transformada de fourier de la señal 1
    :param fourier_R1:  transformada de fourier de la señal 2
    :return:
    norm_cs_R1_Dst1 = modulo normalizado del cross spectrum
    freq_cs_Dst1= frecuencia asociada al módulo del cross spectrum
    phasefreq_R1_Dst1= frecuencia para graficar la fase
    phase_R1_Dst1= fase del cross spectum
    mod_cs_R1_Dst1= módulo del cross spectrum
'''

    mod_Dst1 = np.abs(fourier_Dst1)
    ps_Dst1 = mod_Dst1**2
    mod_Dst1 = np.abs(fourier_R1)
    ps_R1 = mod_Dst1**2


    complexcs_R1_Dst1 = fourier_Dst1 * np.conjugate(fourier_R1)
    mod_cs_R1_Dst1 = np.abs(complexcs_R1_Dst1)
    norm_cs_R1_Dst1 = mod_cs_R1_Dst1 / np.sqrt(ps_Dst1 * ps_R1)
    freq_cs_Dst1 = np.fft.fftfreq(norm_cs_R1_Dst1.size, )

    phase_R1_Dst1 = np.arctan(np.imag(complexcs_R1_Dst1) / np.real(complexcs_R1_Dst1))
    phasefreq_R1_Dst1 = np.fft.fftfreq(phase_R1_Dst1.size, )

    return norm_cs_R1_Dst1, freq_cs_Dst1, phasefreq_R1_Dst1, phase_R1_Dst1, mod_cs_R1_Dst1





# Grafica módulo cross spectrum

def modcsgraph(titulo1, mod_cs_Rl1_Dstl1, freq_cs_Dstl1, titulo2, mod_cs_Rl2_Dstl2, freq_cs_Dstl2):

    fig, axes = plt.subplots(nrows=1, ncols= 2, figsize= (16, 10))

    y1 = mod_cs_Rl1_Dstl1[freq_cs_Dstl1>0]
    x1 = 24 * freq_cs_Dstl1[freq_cs_Dstl1>0]

    axes[0].plot(x1, y1)
    axes[0].set_title(titulo1)
    axes[0].set_ylabel('Cross spectrum')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Frecuencia (1/día)')
    axes[0].set_xscale('log')
    axes[0].grid()

    y2 = mod_cs_Rl2_Dstl2 [freq_cs_Dstl2>0]
    x2 = 24 * freq_cs_Dstl2 [freq_cs_Dstl2>0]

    axes[1].plot(x2,y2)
    axes[1].set_title(titulo2)
    axes[1].set_ylabel('Cross spectrum')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Frecuencia (1/día)')
    axes[1].set_xscale('log')
    axes[1].grid()


def phasecsgraph(titulo1, phasefreq_R1_Dst1, phase_R1_Dst1, titulo2, phasefreq_R2_Dst2, phase_R2_Dst2):
    fig, axes = plt.subplots(nrows=1, ncols= 2, figsize= (16, 10))

    y1_2 = phase_R1_Dst1
    x1_2 = phasefreq_R1_Dst1

    axes[0].plot(x1_2, y1_2, '.')
    axes[0].set_title(titulo1)
    axes[0].set_ylabel('Cross spectrum phase (radians)')
    axes[0].set_xlabel('Frecuencia (1/h)')
    axes[0].set_xscale('log')
    axes[0].grid()

    y2_2 = phase_R2_Dst2[phasefreq_R2_Dst2>0]
    x2_2 = phasefreq_R2_Dst2 [phasefreq_R2_Dst2>0]

    axes[1].plot(x2_2, y2_2,'.')
    axes[1].set_title(titulo2)
    axes[1].set_ylabel('Cross spectrum phase (radians)')
    axes[1].set_xlabel('Frecuencia (1/h)')
    axes[1].set_xscale('log')
    axes[1].grid()




#psgraph('Solar Wind proton density', fourierdic['fourier_SW'].get('hi1'), fourierdic['ffreq_SW'].get('hi1'), fourierdic['fourier_SW'].get('hi2'), fourierdic['ffreq_SW'].get('hi2'), fourierdic['fourier_SW'].get('low1'), fourierdic['ffreq_SW'].get('low1'), fourierdic['fourier_SW'].get('low2'), fourierdic['ffreq_SW'].get('low2') )

norm_cs_R1_SW1, freq_cs_SW1, phasefreq_R1_SW1, phase_R1_SW1, mod_cs_R1_SW1 = crossspectrum(fourierdic['fourier_SW'].get('hi1'), fourierdic['fourier_R'].get('hi1'))
norm_cs_R1_SW1low, freq_cs_SW1low, phasefreq_R1_SW1low, phase_R1_SW1low, mod_cs_R1_SW1low = crossspectrum(fourierdic['fourier_SW'].get('low1'), fourierdic['fourier_R'].get('low1'))

modcsgraph('Nº Sunspots - SW proton density Cross Spectrum Max. 1', mod_cs_R1_SW1, freq_cs_SW1,
           'Nº Sunspots - SW proton density Cross Spectrum Min. 1', mod_cs_R1_SW1low, freq_cs_SW1low)

phasecsgraph('Nº Sunspots - SW proton density Cross Spectrum Phase Max. 1', phasefreq_R1_SW1, phase_R1_SW1,
             'Nº Sunspots - SW proton density Cross Spectrum Phase Min. 1', phasefreq_R1_SW1low, phase_R1_SW1low)



norm_cs_R1_AE1, freq_cs_AE1, phasefreq_R1_AE1, phase_R1_AE1, mod_cs_R1_AE1 = crossspectrum(fourierdic['fourier_AE'].get('hi1'), fourierdic['fourier_R'].get('hi1'))
norm_cs_R1_AE1low, freq_cs_AE1low, phasefreq_R1_AE1low, phase_R1_AE1low, mod_cs_R1_AE1low = crossspectrum(fourierdic['fourier_AE'].get('low1'), fourierdic['fourier_R'].get('low1'))

modcsgraph('Nº Sunspots - AE Index Cross Spectrum Max. 1', mod_cs_R1_AE1, freq_cs_AE1,
           'Nº Sunspots - AE Index Cross Spectrum Min. 1', mod_cs_R1_AE1low, freq_cs_AE1low)

phasecsgraph('Nº Sunspots - AE Index Cross Spectrum Phase Max. 1', phasefreq_R1_AE1, phase_R1_AE1,
             'Nº Sunspots - AE Index Cross Spectrum Phase Min. 1', phasefreq_R1_AE1low, phase_R1_AE1low)



norm_cs_R1_Dst1, freq_cs_Dst1, phasefreq_R1_Dst1, phase_R1_Dst1, mod_cs_R1_Dst1 = crossspectrum(fourierdic['fourier_Dst'].get('hi1'), fourierdic['fourier_R'].get('hi1'))
norm_cs_R1_Dst1low, freq_cs_Dst1low, phasefreq_R1_Dst1low, phase_R1_Dst1low, mod_cs_R1_Dst1low = crossspectrum(fourierdic['fourier_Dst'].get('low1'), fourierdic['fourier_R'].get('low1'))

modcsgraph('Nº Sunspots - Dst Index Cross Spectrum Max. 1', mod_cs_R1_Dst1, freq_cs_Dst1,
           'Nº Sunspots - Dst Index Cross Spectrum Min. 1', mod_cs_R1_Dst1low, freq_cs_Dst1low)

phasecsgraph('Nº Sunspots - Dst Index Cross Spectrum Phase Max. 1', phasefreq_R1_Dst1, phase_R1_Dst1,
             'Nº Sunspots - Dst Index Cross Spectrum Phase Min. 1', phasefreq_R1_Dst1low, phase_R1_Dst1low)



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

pickle.dump(fourierdic, open('DATA/fourierdata', 'wb'))'''
