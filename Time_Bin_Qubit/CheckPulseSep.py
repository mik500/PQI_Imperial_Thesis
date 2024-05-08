import time
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np


dir='C:\\Users\\LocalAdmin\\Imperial College London\\UQOG Quantum Memories - PH - General'

datatype = 'TimeBinQubit' # Type of data e.g. EfficiencyVsPower, used for folder and filename to save data




folder = dir + '\\Data\\2024-03\\2024-03-04_TimeBinQubit\\'

timestamp = '2024-03-04_170125' # interferometer
timestamp2 = '2024-03-04_161639' # AWG Pulses

filename = folder + '\\\\'+ datatype + '_' + timestamp  + '.h5'
filename2 = folder + '\\\\'+ datatype + '_' + timestamp2  + '.h5'


h5 = h5py.File(filename,'r')
data=h5['outputdata'][:]
h5.close()



h5 = h5py.File(filename2,'r')
data2=h5['outputdata'][:]
h5.close()

ch_1 = 2 # Channel for signal
ch_trig = 0 # Channel for trigger
binwidth = 8 # in ps
n_bins = int(250000/binwidth)



t=binwidth/1000*np.arange(n_bins)
 



def double_gauss_function(x, a1, a2, x01, x02, sigma,c):
    return a1*np.exp(-(x-x01)**2/(2*sigma**2)) + a2*np.exp(-(x-x02)**2/(2*sigma**2)) + c

# program
from scipy.optimize import curve_fit

p0 = [2500,2000,55,67.5,0.3,0]
p1 = [1500,1000,105,117.5,0.3,0]

popt, pcov = curve_fit(double_gauss_function, t,data,p0)

popt2, pcov2 = curve_fit(double_gauss_function, t,data2,p1)

fit = double_gauss_function(t,*popt)

fit2 = double_gauss_function(t, *popt2)

plt.figure(1)
plt.clf()
plt.cla()
plt.plot(t-popt[2],data/popt[1], label = 'Data Interferometer')
plt.plot(t-popt2[2],data2/popt2[1], label = 'Data w/o interferometer')
plt.plot(t-popt[2],fit/popt[1], label = 'Fit interferometer data')
plt.plot(t-popt2[2], fit2/popt2[1], label = 'Fit w/o interferometer data')
plt.xlim(-5,15)
plt.legend()
plt.show()


pulse_sep = popt[3] -  popt[2] 
pulse_sep2 = popt2[3] -  popt2[2] 

print('Pulse seperation for interferometer',pulse_sep)
print('Pulse seperation for pulses',pulse_sep2)
print('\n')
difference = np.abs(pulse_sep-pulse_sep2) * 1000
print('Difference in ps', difference)
print('\n')


E1 = double_gauss_function(t,1,0,100,100,popt[4]/2,0)
E2 = double_gauss_function(t,1,0,100.025,100,popt[4]/2,0)

plt.figure(2)
plt.clf()
plt.cla()
plt.plot(t,E1)
plt.plot(t,E2)
plt.xlim(99,101)



overlap = (np.sum(E1 * E2)) / (np.sqrt(np.sum(E1*E1)) * np.sqrt(np.sum(E2*E2)))

print('Overlap', overlap)
#%%


