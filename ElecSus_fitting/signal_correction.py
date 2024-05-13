from mpl_toolkits.mplot3d import axes3d # 3D figure
import matplotlib.pyplot as plt



from matplotlib import cm
from matplotlib import rc
import numpy as np
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import math
import scipy.interpolate as interpolate
import scipy.signal as signal
from scipy.optimize import curve_fit

from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as tck
import scipy.io as sio
import pandas as pd
import matplotlib.ticker as ticker

import os

from pylab import *
from numpy import *

import sys

import glob
import csv
params={'axes.labelsize':20,'xtick.labelsize':15,'ytick.labelsize':15}
rcParams.update(params)

# Colour Palette
rcParams["axes.prop_cycle"]=cycler(color=['#68246D', '#FFD53A', '#00AEEF', '#BE1E2D', '#AFA961'])


### For normalisation


Rb_spectrum_reference = np.loadtxt('ElecSus_fitting/Data/Pumping/Pulsed Pumping/AdjustedBeamWidth/SigCorr_100524_1.csv', delimiter=',', skiprows=2)
signal_correction = np.loadtxt('ElecSus_fitting/Data/Pumping/Pulsed Pumping/AdjustedBeamWidth/SigCorr_100524_3.csv', delimiter=',', skiprows=2)

# Rb_spectrum_reference = np.flipud(Rb_spectrum_reference)
# Rb_spectrum_memory = np.flipud(Rb_spectrum_memory)
Time = signal_correction[:,0]


# Smoothen signal_correction
signal_correction_smooth = signal.savgol_filter(signal_correction[:,1], 51, 3)



# Interpolate signal_correction
signal_correction_fit = interpolate.interp1d(Time, signal_correction_smooth, kind='cubic')(Time)



# Plot data
plt.figure(figsize=(10,5))
plt.plot(Time, signal_correction_smooth, label='Signal correction')
plt.plot(Time, signal_correction_fit, label='Fit')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()

# Save interpolated data
np.savetxt('ElecSus_fitting/Data/Pumping/Pulsed Pumping/AdjustedBeamWidth/SigCorr_100524_3_interp.csv', np.column_stack((Time, signal_correction_fit)), delimiter=',', header='Time (s), Voltage (V)', comments='')




