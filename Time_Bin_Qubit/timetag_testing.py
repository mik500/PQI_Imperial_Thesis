import TimeTagger as ttagger
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import TimeTag
import Visibility as vis

#%%  Parameters

ch_1 = 2 # Channel for signal
ch_trig = 0 # Channel for trigger
binwidth = 16 # in ps
n_bins = int(2500000/binwidth)

t_integration = 20


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
# filename = folder + os.sep +timestamp
TimeTag.save_timetags(ch_det=ch_1, ch_AWG=ch_trig, t_integration=t_integration)


 #%%

swabian = ttagger.createTimeTagger()
swabian.setTriggerLevel(ch_1,0.1)
swabian.setTriggerLevel(ch_trig,0.3)
print('--- Swabian Initialised --- \n')

#%% Save time tags

filename = folder + os.sep +timestamp


histogram = ttagger.Histogram(swabian, ch_1, ch_trig, binwidth = binwidth, n_bins = n_bins)

filewriter = ttagger.FileWriter(tagger=swabian,filename=filename,channels=[ch_1, ch_trig])

time.sleep(t_integration)

filewriter.stop()

outputdata = histogram.getData()
tdata = histogram.getIndex()





ttagger.freeTimeTagger(swabian)

#%% Save histogram

filename_hist = filename + '.csv'

data_hist =np.transpose([tdata,outputdata])
# data = np.array(tdata,outputdata)


np.savetxt(filename_hist,data_hist,delimiter=',')

#%% Plot histogram
plt.figure(1)
plt.clf()
plt.plot(tdata,outputdata)

#%% Import timetag data

# folder = 'C:\\Users\\LocalAdmin\\Imperial College London\\UQOG Quantum Memories - PH - General\\Data\\2023-12\\2023-12-07_TimeBinQubit'
# filename = folder + os.sep + '2023-12-07_154619' #'2023-11-30_162552'
file = filename + '.ttbin'

channels, timestamps = TimeTag.load_tt_data(file)



#%% Import histogram data
file_hist = filename + '.csv'
data_hist = np.loadtxt(file_hist,skiprows=2,delimiter = ',',dtype = float)

t_data_hist = data_hist[:,0]
y_data_hist = data_hist[:,1]

# #%% Always start with first click as trigger

# while channels[0] == ch_1:
#     channels = channels[1:]


#%%

t_start = timestamps[0]
timestamps_rel = timestamps - t_start
t_end = timestamps_rel[-1]

#%% Plot count rate over time

# counter_dur_ms = 10

# counter_dur_ps = counter_dur_ms * 1e9

# num_counters = int(t_end / counter_dur_ps)-1

# clicks_ch1 = np.zeros([num_counters,1])
# clicks_ch_trig = np.zeros([num_counters,1])
# for i in range(num_counters):
#     t_start_range = t_start + (i-1)*counter_dur_ps
#     t_end_range = t_start + i*counter_dur_ps
#     indices = np.where(np.logical_and(timestamps_rel<t_end_range,timestamps_rel>t_start_range))
#     clicks = channels[indices]
#     clicks_ch1[i]=len(np.where(clicks==ch_1)[0])
#     clicks_ch_trig[i]=len(np.where(clicks==ch_trig)[0])

# time_counter = counter_dur_ms*1e-3 * np.arange(num_counters)

# clicks_ch1_Hz=clicks_ch1 * 1e3/counter_dur_ms
# clicks_ch_trig_Hz=clicks_ch_trig * 1e3/counter_dur_ms

# plt.figure(2)
# plt.clf()
# plt.plot(time_counter,clicks_ch1_Hz,label = 'Channel 1')
# plt.plot(time_counter,clicks_ch_trig_Hz,label = 'Trigger')
# plt.xlabel('Time (s)')
# plt.ylabel('Counts (Hz)')
#%% Build histogram (rough method)

time_max = 2000000 # in ps

times = []
for i,ch in enumerate(channels):
    if ch == ch_1:
        if channels[i-1] == ch_trig:
            rel_time = timestamps_rel[i] - timestamps_rel[i-1]
        elif channels[i-2] == ch_trig:
            rel_time = timestamps_rel[i] - timestamps_rel[i-2]
        if rel_time < time_max:
            times.append(rel_time)


#%% Build histogram (better method)

trigNum = np.cumsum(channels == ch_trig) -1

trigTimes = timestamps[np.where(channels == ch_trig)]

start_trigTimes = trigTimes[trigNum]

aligned_times = np.array(timestamps - start_trigTimes)

t_ch1 = aligned_times[np.where(channels == ch_1)]


#%% Make histogram

# binwidth = binwidth
time_max = 2000000 # in ps

n_bins_ttag = int(time_max/binwidth)

counts,bins = np.histogram(times,bins = n_bins_ttag)
# counts,bins = np.histogram(t_ch1,bins = n_bins_ttag)

t = binwidth/1000*np.arange(n_bins_ttag)
plt.figure(2)
plt.clf()
plt.cla()
# plt.hist(bins[:-1], bins, weights=counts)
# plt.stairs(counts, bins)
plt.plot(t,counts)
plt.xlabel('Time (ns)')

plt.show()

#%% Stack subsequent pulses on top of each other

laser_rep = 80.14156e6
t_rep = 1/laser_rep
t_sep = 10*t_rep * 10**12 # separation of pulses from AWG in ps

times_stacked = t_ch1 % t_sep 

time_max = t_sep # in ps

n_bins_ttag = int(time_max/binwidth)

counts2,bins2 = np.histogram(times_stacked,bins = n_bins_ttag)

# bins2 = bins % t_sep


t = binwidth/1000*np.arange(n_bins_ttag)
plt.figure(3)
plt.clf()
plt.cla()
plt.plot(t,counts2)
plt.xlabel('Time (ns)')
# plt.stairs(counts2, bins2)

plt.show()


#%% Create virtual time tagger build histogram from time tags

folder = 'C:\\Users\\LocalAdmin\\Imperial College London\\UQOG Quantum Memories - PH - General\\Data\\2023-12\\2023-12-07_TimeBinQubit'
filename = folder + os.sep + '2023-12-07_154619' #'2023-11-30_162552'
file = filename + '.ttbin'

virtual_tagger = ttagger.createTimeTaggerVirtual()
histogram = ttagger.Histogram(virtual_tagger, ch_1, ch_trig, binwidth = binwidth, n_bins = n_bins)
virtual_tagger.replay(file=file)
time.sleep(1)
virtual_tagger.waitForCompletion(timeout=0)
outputdata_ttags = histogram.getData()

tdata_ttags = histogram.getIndex()

ttagger.freeTimeTagger(virtual_tagger)

#%% Plot things

plt.figure(1)
plt.clf()
plt.cla()

plt.plot(t_data_hist ,y_data_hist)

plt.plot(tdata_ttags,outputdata_ttags)

plt.plot(t*1e3,counts)

plt.xlim(0,100000)
plt.show()
#%%



index_trig = np.where(channels==ch_trig)[0]
times_trig = timestamps[index_trig]
average_trigger_delay = np.mean(np.diff(times_trig))



#%%

# aligned_times_ch1 = vis.align_pulses(channels, timestamps, ch_trig)
# t_ch1 = aligned_times[np.where(channels == ch_1)]

t_hist, counts_hist = vis.build_histogram(channels, timestamps, ch_trig, ch_1, binwidth)


plt.figure(3)
plt.clf()
plt.cla()
plt.plot(t_hist,counts_hist)

plt.show()

