import TimeTagger as ttagger
import time
import h5py
import os
from ImportAndPlot import plot_func
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np

dir='C:\\Users\\LocalAdmin\\Imperial College London\\UQOG Quantum Memories - PH - General'

datatype = 'TimeBinQubit' # Type of data e.g. EfficiencyVsPower, used for folder and filename to save data

t_integration = 2 # Integration time per setting in s for memory measurements


# Swabian parameters
ch_1 = 2 # Channel for signal
ch_trig = 0 # Channel for trigger
binwidth = 16 # in ps
n_bins = int(2500000/binwidth)


#%%  Initialise Swabian

swabian = ttagger.createTimeTagger()
swabian.setTriggerLevel(ch_1,0.1)
swabian.setTriggerLevel(ch_trig,0.3)
print('--- Swabian Initialised --- \n')


#%% 

histogram = ttagger.Histogram(swabian, ch_1, ch_trig, binwidth = binwidth, n_bins = n_bins)
time.sleep(t_integration)
ydata = histogram.getData()
tdata = histogram.getIndex()

plt.figure(1)
plt.clf()
plt.cla()
plt.plot(tdata,ydata)


#%%
# folder = 'C:\\Users\\LocalAdmin\\Imperial College London\\UQOG Quantum Memories - PH - General\\Data\\2024-02\\2024-02-28_TimeBinQubit\\'
# file = 'TimeBinQubit_2024-02-28_115408_memory'

# filename = folder + file  + '.h5'
# h5 = h5py.File(filename,'r')
# ydata=h5['outputdata'][:]
# h5.close()
# tdata=binwidth/1000*np.arange(n_bins)


plt.figure(1)
plt.clf()
plt.cla()
plt.plot(tdata,ydata)

s_max = max(ydata)
idx, pks = scipy.signal.find_peaks(ydata,height=0.05*s_max,distance = 1000)
first_peak_idx = idx[0]

delay_in_ps = tdata[first_peak_idx]

ttagger.setDelayHardware(ch_1,delay_in_ps)


#%%

histogram = ttagger.Histogram(swabian, ch_1, ch_trig, binwidth = binwidth, n_bins = n_bins)
time.sleep(t_integration)
ydata = histogram.getData()
tdata = histogram.getIndex()

plt.figure(2)
plt.clf()
plt.cla()
plt.plot(tdata,ydata)


#%%

ttagger.freeTimeTagger(swabian)
