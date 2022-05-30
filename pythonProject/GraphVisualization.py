#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
#from graphviz import Digraph
import matplotlib.pyplot as plt
import pickle

tedic = pickle.load(open('DATA/tedic', 'rb'))


sp_dst = np.array(tedic['te_sp_b5'])
dst_sp = np.array(tedic['te_b_sp5'])



y1 = sp_dst
y2 = dst_sp
y3 = y1 - y2
x = np.arange(0, 2499, 1)



lines = plt.plot(x, y1, x, y2, x, y3)
plt.setp(lines[0], linewidth=2)
plt.setp(lines[1], linewidth=2)
plt.setp(lines[2], markersize=2)

plt.legend(('TE sp → b', 'TE b → sp', 'TE(sp→b) - TE(b→sp)'),
           loc='upper right')
plt.title('TE High Intensity Window 5')
plt.xlabel('Hours')
plt.ylabel('Transfer Entropy')
plt.show()
