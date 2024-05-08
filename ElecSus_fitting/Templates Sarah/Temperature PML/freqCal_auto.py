import numpy as np
import pandas as pd
import glob
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import sys
import extractTemperature_auto as ET
import matplotlib.pyplot as plt
import xlrd
from openpyxl import Workbook
from scipy import fftpack
#xlrd.xlsx.ensure_elementtree_imported(False, None) #xlsx no longer supported
#xlrd.xlsx.Element_has_iter = True

_Rb87F1 = 4.15; _Rb85F1 = 1.675; _Rb85F2 = -1.232; _Rb87F2 = -2.366
#%% Read Data file    

def readData(file, smooth):
    #open data
    data = pd.read_csv(file)
    time_data = np.asarray(data.Time)
    y_data_ch1 = np.asarray(data.Chan1)
    y_data_ch3 = np.asarray(data.Chan3)
    
    #smooth data
    if(smooth==1):
        window_length = round(0.001*len(time_data))+1;polyorder = 1
        y_data_ch1 = savgol_filter(y_data_ch1, window_length, polyorder)
        y_data_ch3 = savgol_filter(y_data_ch3, window_length, polyorder)
    else:
        pass
    return time_data, y_data_ch1, y_data_ch3

def readDataICL(file, smooth, format):
    # if format == 'csv':
    #     wb = pd.read_csv(file, skiprows= 2, names = ['time', 'ch1', 'ch2', 'ch3'])
    #     y_data_ch1 = np.asarray(wb['ch1'])
    #     y_data_ch2 = np.asarray(wb['ch2'])
    #     y_data_ch3 = np.asarray(wb['ch3'])
    #     time_data = np.arange(0,len(y_data_ch1))
    # elif format == 'xls':
    #     pd = pd.read_excel(file)
    #     pd.columns = ['ch1', 'ch2']
    #     y_data_ch1 = np.asarray(pd['ch1'])
    #     y_data_ch2 = np.asarray(pd['ch2'])
    #     time_data = np.arange(0,len(y_data_ch1))

    # wb = pd.read_csv(file, skiprows= 2, names = ['time', 'ch1', 'ch2', 'ch3'])
    # y_data_ch1 = np.asarray(wb['ch1'])
    # y_data_ch2 = np.asarray(wb['ch2'])
    # y_data_ch3 = np.asarray(wb['ch3'])
    # time_data = np.arange(0,len(y_data_ch1))

    wb = pd.read_excel(file)
    wb.columns = ['ch1', 'ch2']
    y_data_ch1 = np.asarray(wb['ch1'])
    y_data_ch2 = np.asarray(wb['ch2'])
    time_data = np.arange(0,len(y_data_ch1))


    
    #smooth data
    if(smooth==1):
        # fu = 0.2
        # fc=0.05
        # fw = 0.01
        # y_data_ch1 = fftFilt(y_data_ch1,fu,fc,fw)
        # y_data_ch3 = fftFilt(y_data_ch3,fu,fc,fw)
        if format == 'xls':
            window_length = 3;polyorder = 1
            y_data_ch1 = savgol_filter(y_data_ch1, window_length, polyorder)
            y_data_ch2 = savgol_filter(y_data_ch2, window_length, polyorder)

        if format == 'csv':
            window_length = 3;polyorder = 1
            y_data_ch1 = savgol_filter(y_data_ch1, window_length, polyorder)
            y_data_ch2 = savgol_filter(y_data_ch2, window_length, polyorder)
            y_data_ch3 = savgol_filter(y_data_ch3, window_length, polyorder)
       
    else:
        pass
    
    if format == 'csv':
        return time_data, y_data_ch1, y_data_ch2, y_data_ch3

    if format == 'xls':
        return time_data, y_data_ch1, y_data_ch2

   




def fftFilt(sig,fu,fc,fw):

    sig_fft = fftpack.fft(sig)
    sample_freq = fftpack.fftfreq(sig.size, d=1)
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > fu] = 0
    high_freq_fft[np.where(np.logical_and(sample_freq<(fc+fw), sample_freq>(fc-fw)))]=0
    sig = fftpack.ifft(high_freq_fft)
    return sig


def getTargetData(timestamp,all_files):
    target_data=[]
    for i in range(0,len(all_files)):
        s = all_files[i]
        if s.find(timestamp) != -1:
            target_data.append(s)
    return target_data


#%%
def normaliseData(time_data,y_data,peak_points):# noramlises the data such that off-resonance is a transmision of 1 i.e. OD of 0. Returns OD
    f_points = np.asarray([_Rb87F2,_Rb87F1])

    
    f_array = np.polyval(np.polyfit([peak_points[0],peak_points[3]],f_points,1),np.asarray(range(0,len(time_data))))
    w=3
    indices = np.where(f_array<(f_points[0]-w))
    indices3 = np.where(f_array>(f_points[1]+w))
    indices = np.append(indices, [indices3])
    
    w0 = .001
    indices_zero = np.where(np.logical_and(f_array>(f_points[0]-w0), f_array<(f_points[0]+w0)))
    y_data = y_data-np.mean(y_data[indices_zero])

    
    p = np.polyfit(time_data[indices], y_data[indices], 3)
    bg = np.polyval(p, time_data)
    return y_data/bg

def normaliseSatSpecData(time_data,y_data, prominence, distance):# noramlises the data such that off-resonance is a transmision of 1 i.e. OD of 0. Returns OD
    # defining the indices for background
    OD = -np.log(y_data/max(y_data))
    peak_points, _ = find_peaks(OD, prominence=prominence, distance=distance)
    print(peak_points)
    
    f_points = np.asarray([_Rb87F1,_Rb85F1,_Rb85F2,_Rb87F2])
    f_array = np.polyval(np.polyfit(peak_points,f_points,1),np.asarray(range(0,len(time_data))))
    w=0.9
    point_A = f_points[0]+w
    point_B = f_points[0]-w
    point_C = f_points[1]+w
    point_D = f_points[1]-w
    point_E = f_points[2]+w
    point_F = f_points[3]-w
    point_G = -4.5

    indices = np.where(f_array>(f_points[0]+w))
    indices2 = np.where(np.logical_and(f_array>(f_points[1]+w), f_array<(f_points[0]-w)))
    indices3 = np.where(np.logical_and(f_array<(f_points[1]-w),f_array>(f_points[2] + w)))
    indices4 = np.where(np.logical_and(f_array<(f_points[3]-w),f_array>(-4.5)))
    indices = np.append(indices, [indices2])
    indices = np.append(indices, [indices3])
    indices = np.append(indices, [indices4])

    plt.plot(f_array, OD)
    plt.plot(f_array[peak_points],OD[peak_points],'x')

    # Abline points
    plt.axvline(x=point_A, color='r', linestyle='--')
    plt.axvline(x=point_B, color='r', linestyle='--')
    plt.axvline(x=point_C, color='r', linestyle='--')
    plt.axvline(x=point_D, color='r', linestyle='--')
    plt.axvline(x=point_E, color='r', linestyle='--')
    plt.axvline(x=point_F, color='r', linestyle='--')
    plt.axvline(x=point_G, color='r', linestyle='--')
    

    
    p = np.polyfit(f_array[indices], y_data[indices], 1)
    bg = np.polyval(p, f_array)
    return y_data/bg, f_array, peak_points

def frequencyCalibrate(time_data,chan1,chan3, RbFrac, cellLength, temp, temp_Dopp, plotFlag, cellFlag, Constrain, prominence, distance):
    un_pumped_satSpec, f_array, peak_points = normaliseSatSpecData(time_data,chan3, prominence, distance)
   
    if(cellFlag==0):
        _signal = un_pumped_satSpec
    elif(cellFlag==1):
        un_pumped = normaliseData(time_data,chan1,peak_points)
        _signal = un_pumped
    else:
        sys.exit('Cell flag is either 0 (ref cell) or 1 (memory cell). Try again.')
    #%% this part does the fitting
    p0=[0, 1, -1e-3,1e-6, RbFrac, temp, temp_Dopp]
    ET.globalVars(cellLength,Constrain) # send through the variables 
    [freq, residuals, temperature, temp_Doppler, Rb85Frac, popt] , [data_signal,_fit,residual, rms, res_max]= ET.extractTemp(f_array, _signal, p0, plotFlag)
    
    return [freq, residuals, temperature, temp_Doppler, Rb85Frac, popt], [data_signal,_fit,residual, rms, res_max]

def getDensity(T):
    k  = 1.380649e-23
    P = getPressure(T)
    density = P/k/(T+ 273.15)
    return density

def getPressure(T):
    TK = T+273.15
    if(T<39.48):
        Pv = 133.322*10**(-94.04826 - 1961.258/(TK)  - 0.03771687*(TK) + 42.57526*np.log10(TK))
    else:
        Pv = 133.322*10**(15.88253 - 4529.635/(TK) + 0.00058663*(TK) -2.99138*np.log10(TK))
    return Pv
        
        
        
        
        
        
        

