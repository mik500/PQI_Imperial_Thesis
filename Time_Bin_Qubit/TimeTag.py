import TimeTagger as ttagger
import numpy as np
import time
import os

def save_timetags(ch_det, ch_AWG, ch_piezo, t_integration):
   
    d = os.getcwd() 
    directory=d[:-38]
    datatype = 'TimeBinQubit' # Type of data e.g. EfficiencyVsPower, used for folder and filename to save data
    t = time.localtime()
    folder_month = directory + 'Data\\' + time.strftime('%Y-%m', t)
    if not os.path.exists(folder_month):
        os.makedirs(folder_month)
     
    folder = folder_month + '\\' + time.strftime('%Y-%m-%d', t) + '_' + datatype
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = time.strftime('%Y-%m-%d_%H%M%S', t)
    print(timestamp[-6:])
    print('\n')  
   
    swabian = ttagger.createTimeTagger()
    swabian.setTriggerLevel(ch_det,0.1)
    swabian.setTriggerLevel(ch_AWG,0.3)
    swabian.setTriggerLevel(ch_piezo,0.3)
    
    filename = folder + os.sep +timestamp
    
    filewriter = ttagger.FileWriter(tagger=swabian,filename=filename,channels=[ch_det, ch_AWG,ch_piezo])
    time.sleep(t_integration)
    filewriter.stop()
   
    ttagger.freeTimeTagger(swabian)

    return filename


    
   
def load_tt_data(filepath):
    filereader = ttagger.FileReader(filepath)
    n_events = 1000000  # Number of events to read at once
    channels = np.array([])
    times = np.array([])
    i = 0
    while filereader.hasData():
        # print('Loading ith chunk of 1000000 elements (or less): ', i)
        data = filereader.getData(n_events=n_events)

        channels_temp = data.getChannels()      # The channel numbers
        times_temp = data.getTimestamps()       # The timestamps in ps
        overflow_types = data.getEventTypes()   # TimeTag = 0, Error = 1, OverflowBegin = 2, OverflowEnd = 3, MissedEvents = 4
        missed_events = data.getMissedEvents()  # The numbers of missed events in case of overflow
        if np.any(missed_events != 0):
            print('Error! Missed events: ', missed_events)
        channels = np.append(channels, channels_temp)
        times = np.append(times, times_temp)
    del filereader
    return channels, times