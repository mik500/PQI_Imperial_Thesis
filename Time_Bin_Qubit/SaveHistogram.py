
import TimeTagger as ttagger
import time
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
dir='C:\\Users\\LocalAdmin\\Imperial College London\\UQOG Quantum Memories - PH - General'

datatype = 'TimeBinQubit' # Type of data e.g. EfficiencyVsPower, used for folder and filename to save data

t_integration =  60# Integration time per setting in s for memory measurements


# Swabian parameters
ch_1 = 2 # Channel for signal
ch_trig = 0 # Channel for trigger
binwidth = 8 # in ps
n_bins = int(250000/binwidth)


#%%  Initialise Swabian

swabian = ttagger.createTimeTagger()
swabian.setTriggerLevel(ch_1,0.1)
swabian.setTriggerLevel(ch_trig,0.3)
print('--- Swabian Initialised --- \n')

#%% Set up folder and timestamp for saving data
t = time.localtime()
folder_month = dir + '\\Data\\' + time.strftime('%Y-%m', t)
if not os.path.exists(folder_month):
    os.makedirs(folder_month)

folder = folder_month + '\\' + time.strftime('%Y-%m-%d', t) + '_' + datatype
if not os.path.exists(folder):
    os.makedirs(folder)
timestamp = time.strftime('%Y-%m-%d_%H%M%S', t)
print(timestamp[-6:])
print('\n') 
   



filename = folder + '\\\\'+ datatype + '_' + timestamp  + '.h5'
histogram = ttagger.Histogram(swabian, ch_1, ch_trig, binwidth = binwidth, n_bins = n_bins)
time.sleep(t_integration)
outputdata = histogram.getData()
tdata = histogram.getIndex()
with h5py.File(filename,'w') as hdf:
     hdf.create_dataset('outputdata',data = outputdata, dtype = 'float64' )
     
     
     
ttagger.freeTimeTagger(swabian)




#%%

t=binwidth/1000*np.arange(n_bins)
 
plt.figure(1)
plt.clf()
plt.cla()
plt.plot(t,outputdata)