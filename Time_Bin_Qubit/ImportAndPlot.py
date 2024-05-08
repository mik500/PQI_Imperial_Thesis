import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import time

#%%
def plot_func(timestm, t_integration, binwidth, n_bins, f1 = None):
# timestamp = '2023-11-30_143615'
# timestm = timestamp[-6:]

    transmission= 0.23 #transmison from cell to detector
    detector_eff = 0.8
    trigger_rate = 500e3
    t_integration = 30 # Integration time of the measurement in seconds
    
    
    window_size = 1.1 # integration window in ns
    storage_time = 24.9 # storage time in ns
    # binwidth = 16 #resolution in ps
    # n_bins = int(2500000/binwidth)
    t=binwidth/1000*np.arange(n_bins)
        
    if f1 is None:
        tt = time.localtime()
        f1 = time.strftime('%Y-%m-%d', tt)
    # f1 = '2023-11-30'
    f2 = 'TimeBinQubit'
    timestamp = str(timestm)   #'125413'
        
    d = os.getcwd()
    
    folder = d[:-38]+'Data\\' + f1[0:7] + '\\'+ f1 + '_' + f2 + '\\'
    file = f2 + '_'+ f1 +'_'+ timestamp + '_'
    
    
    # settings = ['signal','memory','dressing','noise', 'control']
    # settings=['signal','memory','rephasing']
    
    settings=['signal','memory','rephasing']
    # settings=['signal','memory','rephasing','dark','control only','transfer only','control and transfer']
    
    
    data=np.zeros([len(settings),n_bins])
    
    for k in range(len(settings)):
        filename = folder + file + settings[k] + '.h5'
        h5 = h5py.File(filename,'r')
        data[k,:]=h5['outputdata'][:]
        h5.close()
    
    i_sig = settings.index('signal')
    i_mem=settings.index('memory')
    if len(settings) > 2:
        i_reph = settings.index('rephasing') 
    
    if len(settings) > 3:
        i_noise = settings.index('control and transfer')
        i_control = settings.index('control only')
    
      
    #%% Find integration windows
    
    # # find peak
    loc_peaks = np.argmax(data[i_sig, :])
    #t_window = np.array([1000, 1100]) #50ns 
    #loc_peaks = np.argmax(data[i_sig, (t > t_window[0]) & (t < t_window[1]) ]) +  np.argmin(np.abs(t - t_window[0])) #50 ns

    t_input = t[loc_peaks]
    window = window_size* 1000/binwidth
    tin_0 = loc_peaks-int(window/2)
    tin_1 = loc_peaks+int(window/2)  
    
    tout_0 = tin_0 + int(storage_time * 1000/binwidth)
    tout_1  =tin_1 + int(storage_time * 1000/binwidth)
    
   
    
 
    # t_input = 1066
    
    
    # tin_0 = int(1065.4* 1000/binwidth) #stat of input time window 115.6  127
    # tin_1 = int(1067.6* 1000/binwidth) #end of input time window 118
    
    # tout_0 = int(1090.7* 1000/binwidth) #start of output time window 118
    # tout_1 = int(1092.10* 1000/binwidth) #end of output time window 121
    
    #%% Sum counts in integration windows
    
    sum_windows = np.zeros([len(settings),2])
    
    for i in range(len(settings)):
        sum_windows[i,0] = np.sum(data[i,tin_0:tin_1])
        sum_windows[i,1] = np.sum(data[i,tout_0:tout_1])
        
    err_sum_windows = np.sqrt(sum_windows)
    
    
    #%% Calculate efficiencies
    
    if len(settings)<=3: # No noise or rephasing measurement
        readin_efficiency = 1-(sum_windows[i_mem,0])/sum_windows[i_sig,0]
        total_efficiency = (sum_windows[i_mem,1])/sum_windows[i_sig,0]
        
        err_readin_eff = readin_efficiency*np.sqrt((err_sum_windows[i_mem,0])**2/(sum_windows[i_mem,0])**2 + (err_sum_windows[i_sig,0]/sum_windows[i_sig,0])**2)
        err_total_eff = total_efficiency * np.sqrt((err_sum_windows[i_mem,1]**2 )/(sum_windows[i_mem,1] )**2 + (err_sum_windows[i_sig,0]/sum_windows[i_sig,0])**2)
    
    # if len(settings)<=3: # No noise  measurement  
        readin_efficiency_reph = 1-(sum_windows[i_reph,0])/sum_windows[i_sig,0]
        total_efficiency_reph = (sum_windows[i_reph,1])/sum_windows[i_sig,0]
        
    
        err_readin_eff_reph = readin_efficiency_reph*np.sqrt((err_sum_windows[i_reph,0])**2/(sum_windows[i_reph,0])**2 + (err_sum_windows[i_sig,0]/sum_windows[i_sig,0])**2)
        err_total_eff_reph = total_efficiency_reph * np.sqrt((err_sum_windows[i_reph,1]**2 )/(sum_windows[i_reph,1] )**2 + (err_sum_windows[i_sig,0]/sum_windows[i_sig,0])**2)
    
    
    if len(settings)>3: # With noise measurement
        
        readin_efficiency = 1-(sum_windows[i_mem,0]-sum_windows[i_control,0])/sum_windows[i_sig,0]
        total_efficiency = (sum_windows[i_mem,1]-sum_windows[i_control,1])/sum_windows[i_sig,0]
        
        readin_efficiency_reph = 1-(sum_windows[i_reph,0]-sum_windows[i_noise,0])/sum_windows[i_sig,0]
        total_efficiency_reph = (sum_windows[i_reph,1]-sum_windows[i_noise,1])/sum_windows[i_sig,0]
        
        
        err_readin_eff = readin_efficiency*np.sqrt((err_sum_windows[i_mem,0]+err_sum_windows[i_control,0])**2/(sum_windows[i_mem,0]-sum_windows[i_control,0])**2 + (err_sum_windows[i_sig,0]/sum_windows[i_sig,0])**2)
        err_total_eff = total_efficiency * np.sqrt((err_sum_windows[i_mem,1]**2 + err_sum_windows[i_control,1]**2)/(sum_windows[i_mem,1] - sum_windows[i_control,1])**2 + (err_sum_windows[i_sig,0]/sum_windows[i_sig,0])**2)
        
        err_readin_eff_reph = readin_efficiency_reph*np.sqrt((err_sum_windows[i_reph,0]+err_sum_windows[i_noise,0])**2/(sum_windows[i_reph,0]-sum_windows[i_noise,0])**2 + (err_sum_windows[i_sig,0]/sum_windows[i_sig,0])**2)
        err_total_eff_reph = total_efficiency_reph * np.sqrt((err_sum_windows[i_reph,1]**2 + err_sum_windows[i_noise,1]**2)/(sum_windows[i_reph,1] - sum_windows[i_noise,1])**2 + (err_sum_windows[i_sig,0]/sum_windows[i_sig,0])**2)
    
    
    print('Read in eff (no dressing) = ' , format(readin_efficiency,'.4f'),'±', format(err_readin_eff,'.4f'))
    print('Total eff (no dressing) = ' ,format(total_efficiency,'.4f'),'±',format(err_total_eff,'.4f'))
    print('Read in eff dressing = ' , format(readin_efficiency_reph,'.4f'),'±', format(err_readin_eff_reph,'.4f'))
    print('Total eff dressing = ' ,format(total_efficiency_reph,'.4f'),'±',format(err_total_eff_reph,'.4f'))
    
    
    
    #%% Noise and SNR calculations
    
    if len(settings)>3:
        SNR = (sum_windows[i_mem,1] - sum_windows[i_control,1]) / (sum_windows[i_control,1])
        SNR_reph = (sum_windows[i_reph,1] - sum_windows[i_noise,1]) / (sum_windows[i_noise,1])
    
        input_photons_per_pulse = sum_windows[i_sig,0]/t_integration/trigger_rate/transmission/detector_eff
        err_input_photons_per_pulse = err_sum_windows[i_sig,0]/t_integration/trigger_rate/transmission/detector_eff
        
        noise_photons_per_pulse_output = sum_windows[i_noise,1]/t_integration/trigger_rate/transmission/detector_eff
        err_noise_photons_per_pulse_output = err_sum_windows[i_noise,1]/t_integration/trigger_rate/transmission/detector_eff
    
        control_photons_per_pulse_output = sum_windows[i_control,1]/t_integration/trigger_rate/transmission/detector_eff
        err_control_photons_per_pulse_output = err_sum_windows[i_control,1]/t_integration/trigger_rate/transmission/detector_eff
    
        # mu_1= control_photons_per_pulse_output / total_efficiency
        mu_1_reph= noise_photons_per_pulse_output / total_efficiency_reph
    
    
        print('Input photons per pulse = ',format(input_photons_per_pulse,'.8f'),'+/-',format(err_input_photons_per_pulse,'.8f'))
        print('Control and Transfer Noise photons per pulse = ',format(1E6*noise_photons_per_pulse_output,'.4f'),'+/-',format(1E6*err_noise_photons_per_pulse_output,'.4f'),'x 10^-6')
        print('Control photons per pulse = ',format(1E6*control_photons_per_pulse_output,'.4f'),'+/-',format(1E6*err_control_photons_per_pulse_output,'.4f'),'x 10^-6')
    
        print('SNR (no transfer) = ',format(SNR,'.4f'))
        print('SNR transfer = ',format(SNR_reph,'.4f'))
    
        print('mu1 transfer= ',format(1E6*mu_1_reph,'.4f'),'x 10^-6')
    
    
        
    
    #%% Plot things
    norm = np.max(data[i_sig,:])
    
    colors = ['black','crimson','DarkBlue','DarkOrange','Magenta','Dark Green','Purple']    
    plt.figure(3)
    plt.cla()
    plt.clf()
    plt.fill_between([tin_0*binwidth/1000,tin_1*binwidth/1000]-t_input,0,[1.2*norm,1.2*norm],color='lightgrey')
    plt.fill_between([tout_0*binwidth/1000,tout_1*binwidth/1000]-t_input,0,[1.2*norm,1.2*norm],color='mistyrose')
    for k in range(len(settings)):
        plt.plot(t-t_input,data[k,:],label=settings[k],color=colors[k])
        #plt.plot(t,data[k,:],label=settings[k],color=colors[k])
        
    plt.legend()
    plt.tight_layout()
    # plt.xlim(1065,1110)
    plt.xlim( - 2, 2*storage_time + 2)
    plt.xlabel('Time (ns)')
    plt.show()  
    
    return
