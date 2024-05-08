import sys
import glob
import freqCal_auto as FC # import this
import time

GHz = 1e9
MHz = 1e6
t0 = time.time()
#%% open your favourite data
date_stamp = '20220408'
target_timestamp = "15-56-42"
main_folder = 'datafolder'
path = main_folder +'/' + date_stamp +'/'
all_files = glob.glob(path + "*.csv")
all_files.sort()
try:
    files = FC.getTargetData(target_timestamp,all_files) # calling a function in FC to get the file you care about
    files = files[-1:][0] # get the no pumping case
except:
    sys.exit('Error. Check filename, date and timestamp.')

#%%
smoothFlag = 1 # 0 for raw data, 1 for smoothed data    
plotFlag = 1 # 1 for yes, 0 for no
cellFlag = 1 # 0 for ref cell, 1 for memory cell

# done this as there are these two options in the soton lab. Either you use the ref cell which has natural abundance, of the memory cell which is pure Rb87
# can just do these by hand in the function below
if(cellFlag==1):
    RB85frac = 0 # in percentage
    L = 0.1 # length of cell
elif(cellFlag==0):
    RB85frac = 72.17 # in percentage
    L = 0.075 # length of cell
else:
    sys.exit('Try again')
#%%
# using a function in FC to get the time data, chan1  and chan3. Smooth flag is there is indicate if you want smoothing or not 
time_data, chan1, chan3 = FC.readData(files, smoothFlag) 

temp_init = 20
temp_Dopp = 20
Constrain = False # choose true to have the temp and Doppler temp constrained, false to have as independent parameters

# calling a function in FC to calibrate the frequency axis. This does a basic calibration using the four dips of the reference cell, 
# then recalibrates by fitting a function from elecsus. You can choose to plot the fit, and which cell to use for this, see above.
freq, residuals, temperature, temp_Doppler, RbFrac = FC.frequencyCalibrate(time_data,chan1,chan3, RB85frac, L, temp_init, temp_Dopp, plotFlag, cellFlag, Constrain) 

#%%
print('Temp (Atom Number) =',temperature,'°C')
if(Constrain):
    pass
else:
    print('Temp (Doppler) =',temp_Doppler,'°C')
print('Rb85|Rb87:',RbFrac,'|',round(100-RbFrac,2))


#%% If you already have a frequency calibration and what to fit a temperature curve using elecsus, then use this

# import extractTemperature_auto as ET #import this
# from elecsus_methods import calculate as get_spectra
# import numpy as np
# import matplotlib.pyplot as plt

# p0 = [f0, f1, f2, f3, temp, temp_Dopp, RbFrac] # The 'fs' are the scale adjust to the freq axis as (f3*f**3 + f2*f**2 + f1*f - f0)
# ET.globalVars(cell_length, constrain_temp_fit) # pass through the global params
# _signal = FC.normaliseData(time_data,chan1) # calling a function in FC to normalise the chan1 data
# temperature, residuals, _ = ET.extractTemp(freq, _signal, p0, plot_flag) #calling a function in ET to extract the temperature of the cell

#%%
t1 = time.time()
total = t1-t0
print('Time taken = ',round(total,2),'s')
#%%






