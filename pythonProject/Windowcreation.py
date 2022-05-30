
import re
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft, fftfreq
import pickle



def windowcreate(counter, maxmin, x, y, sigma):
    nsuns_window=[]
    for i in range(0,len(x)):
        if (maxmin[counter]==x[i]):
            nsuns_window.append(y[maxmin[counter]-sigma:maxmin[counter]+sigma])
    return nsuns_window




windowdic = pickle.load(open('DATA/windowdata', 'rb'))

for i in range(1, 5):
    R_max = windowdic['b_low']
    R_max1 = R_max[i]
    Rarray = np.array(R_max1).flatten()


    x = np.arange(0, 20000, 1)
    y = Rarray

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Hours', ylabel='Scalar B, nT',
           title=f'Scalar B: Low Intensity Window {i}')
    ax.grid()
