import numpy as np
import time
import os



def align_triggers(channels, times, trig_ch):
    trigNum = np.cumsum(channels == trig_ch) -1
    trigTimes = times[np.where(channels == trig_ch)]
    start_trigTimes = trigTimes[trigNum]
    times_shifted = np.array(times - start_trigTimes)
    return times_shifted


def build_histogram(channels,times, ch_trig, ch_det,binwidth,laser_rep,n_bins):
    

     
    aligned_times = align_triggers(channels,times,ch_trig)
    aligned_times_ch1 = aligned_times[np.where(channels == ch_det)]
    
    #laser_rep = 80.14156*1e6 # repetition rate of laser
    timebin_sep = 1/laser_rep  # laser pulse separation
    t_rep = 8*timebin_sep * 10**12 # separation time between subsequent experiments (set on AWG code to be 8 x laser pulse separation, or approximately 100ns)

    # times_stacked = aligned_times_ch1 % t_rep # stack all pulses after single AWG trigger on top of each other (instead of appearing every t_rep)

    # time_max = t_rep # in ps   
    time_max = max(aligned_times_ch1)
    # n_bins = int(time_max/binwidth)
    
    # counts,bins = np.histogram(times_stacked,bins = n_bins)
    counts,bins = np.histogram(aligned_times_ch1,bins = n_bins)
    t_hist = binwidth*np.arange(n_bins)
    
    return t_hist, counts


def vis_piezo_scan(channels,times,ch_piezo,ch_det,ch_AWG,binwidth,num_time_windows, num_piezo_scans_to_av, num_phase_bins,t_offset,pulse_duration,laser_rep):
    
    times_piezo = times[np.where(channels==ch_piezo)[0]] # timestamps of piezo triggers
    average_piezo_duration = np.mean(np.diff(times_piezo)) # average time between piezo triggers
    total_num_piezos = len(times_piezo)
    num_piezo_sections = (total_num_piezos-1) // num_piezo_scans_to_av 
    
    ## set regions of interest of pulses after AWG triggers  
    #laser_rep = 80.14156*1e6 # repetition rate of laser
    timebin_sep = 1/laser_rep *1e12 # laser pulse separation


    ROIs = np.zeros((num_time_windows,2))
    
    for j in range(num_time_windows):
        ROIs[j,0] = int(t_offset+j*timebin_sep-pulse_duration/2)*1/binwidth
        ROIs[j,1] = int(t_offset+j*timebin_sep+pulse_duration/2)*1/binwidth
    
    print(ROIs)
    sum_ROIs = np.zeros((num_time_windows, num_piezo_sections,num_phase_bins))

    t_rep = 8*timebin_sep  # separation time between subsequent experiments (set on AWG code to be 8 x laser pulse separation, or approximately 100ns)
    time_max = t_rep # in ps   
    n_bins = int(2000000/binwidth)

    hist_t = np.zeros((num_piezo_sections,num_phase_bins,n_bins))
    hist_y = np.zeros((num_piezo_sections,num_phase_bins,n_bins))

    for k in range(num_piezo_sections):
        t_start_scan = times_piezo[k * num_piezo_scans_to_av ] # starting timestamp of piezo scan of interest
        t_end_scan = times_piezo[(k+1)*num_piezo_scans_to_av] # end timestamp after N piezo scans (we are averaging over a few piezo scans to get better statistics) 

        timestamps_scan = times[np.where((times>t_start_scan)&(times<t_end_scan))[0]] # get timestamps of these N scans
        channels_scan = channels[np.where((times>t_start_scan)&(times<t_end_scan))[0]] # get channel number of these N scans
        times_aligned = align_triggers(channels_scan,timestamps_scan,ch_piezo) # align these timestamps with respect to the piezo trigger (i.e. all piezo triggers now arrive at 0)
        
        times_clipped = times_aligned[np.where((times_aligned>0.1*average_piezo_duration)&(times_aligned<0.4*average_piezo_duration))[0]] # only take timestamps within first half of piezo scan, and more than 10% away from turning points
        channels_clipped = channels_scan[np.where((times_aligned>0.1*average_piezo_duration)&(times_aligned<0.4*average_piezo_duration))[0]]

        t_start = min(times_clipped)
        t_end = max(times_clipped)
        t_duration = t_end - t_start

        for i in range(num_phase_bins): # split piezo scan into a certain number of phase bins
            t_start_phasebin = t_start +  i/num_phase_bins*t_duration # starting timestamp of this phase bin
            t_end_phasebin = t_start +  (i+1)/num_phase_bins*t_duration # ending timestamp of this phase bin
            index_bin = np.where((times_clipped>t_start_phasebin)&(times_clipped<t_end_phasebin))[0] 
            times_bin = times_clipped[index_bin] # get timestamps within this phase bin
            ch_bin = channels_clipped[index_bin] # get channel numbers within this phase bin
    
            t_hist, counts = build_histogram(ch_bin, times_bin, ch_AWG, ch_det, binwidth,laser_rep,n_bins) # build a histogram between the AWG trigger and the detector in this phase bin

            hist_t[k,i,:] = t_hist
            hist_y[k,i,:] = counts

            sum_ROIs[j,k,i] = np.sum(counts[int(ROIs[j,0]):int(ROIs[j,1])])# record how many counts are in each region of interest

    
    return sum_ROIs,hist_t,hist_y


    
    