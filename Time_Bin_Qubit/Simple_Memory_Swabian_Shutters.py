
import TimeTagger as ttagger
import time
import h5py
import os
import shutters as shutters
from ImportAndPlot import plot_func


dir='C:\\Users\\LocalAdmin\\Imperial College London\\UQOG Quantum Memories - PH - General'

datatype = 'TimeBinQubit' # Type of data e.g. EfficiencyVsPower, used for folder and filename to save data

t_integration = 10 # Integration time per setting in s for memory measurements

#Choose from: 
    #'signal'  = signal only
    #'memory' = signal and control
    #'rephasing' = signal and control and transfer
    #'dark' = everything blocked
    #'control only'
    #'transfer only'
    #'control and transfer'

settings=['signal','memory','rephasing']
# settings=['signal','memory','rephasing','dark','control only','transfer only','control and transfer']


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

#%%  Initialise Shutters
shutterFlag = 1

board, servos =shutters.initialise_shutters(shutterFlag)
sA,CA,TA = shutters.find_angles(settings)

print('--- Shutters Initialised --- \n')


#%% definitions
def saveHist(timestamp, t_integration,setting):
    filename = folder + '\\\\'+ datatype + '_' + timestamp + '_' + setting + '.h5'
    histogram = ttagger.Histogram(swabian, ch_1, ch_trig, binwidth = binwidth, n_bins = n_bins)
    time.sleep(t_integration)
    outputdata = histogram.getData()
    with h5py.File(filename,'w') as hdf:
         hdf.create_dataset('outputdata',data = outputdata, dtype = 'float64' )
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
   
#%% Take data 
for k in range(len(settings)):
        
        
    if(shutterFlag):
        shutters.go_to_angle(servos, [sA[k],CA[k],TA[k]])

    print("Taking Data",settings[k])
    saveHist(timestamp, t_integration,settings[k])
    print('Data taking done \n')
                    


#%% Open all shutters at end of measurement    
if(shutterFlag):
    shutters.go_to_angle(servos, [90,90,90])
    
#%% Close things
# if(shutterFlag):
board.exit()  

ttagger.freeTimeTagger(swabian)

#%% Plot and analyse data
plot_func(timestamp[-6:], t_integration = t_integration,binwidth = binwidth, n_bins = n_bins)
