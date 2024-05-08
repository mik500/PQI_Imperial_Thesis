

import TimeTagger as ttagger
import time
import os


#%%  Parameters

ch_piezo = 3 # Channel for Piezo
ch_1 = 2 # Channel for signal
ch_trig = 0 # Channel for trigger
binwidth = 16 # in ps
n_bins = int(2500000/binwidth)

t_integration = 10


#%% Set up folder 
dir='C:\\Users\\LocalAdmin\\Imperial College London\\UQOG Quantum Memories - PH - General'
datatype = 'TimeBinQubit' # Type of data e.g. EfficiencyVsPower, used for folder and filename to save data


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



 #%%

swabian = ttagger.createTimeTagger()
swabian.setTriggerLevel(ch_1,0.1)
swabian.setTriggerLevel(ch_trig,0.3)
swabian.setTriggerLevel(ch_piezo,0.3)
print('--- Swabian Initialised --- \n')

#%% Save time tags

filename = folder + os.sep +timestamp


filewriter = ttagger.FileWriter(tagger=swabian,filename=filename,channels=[ch_1, ch_trig, ch_piezo])

time.sleep(t_integration)

filewriter.stop()


ttagger.freeTimeTagger(swabian)


