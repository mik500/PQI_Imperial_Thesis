# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:41:49 2023

@author: LocalAdmin
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, savgol_filter
import extractTemperature_auto as ET
cwd = os.getcwd()
import definitions as defs
from elecsus.libs import numberDensityEqs as n
from scipy.constants import physical_constants, c


def envelope_func(x):
    y = -0.0000001*x**3 + 100
    return y
    

x_data = np.array(range(0,100))
y_data = envelope_func(x_data)
print (y_data)
plt.plot(x_data,y_data)
plt.ylim()
plt.show()