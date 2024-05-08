import matplotlib.pyplot as plt
import numpy as np
from elecsus_methods import calculate as get_spectra
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import colors as mcolors
import os
GHz = 1e9
MHz = 1e6
#%%
def cc(arg,a):
    return mcolors.to_rgba(arg, alpha=a)

def globalVars(A,B):
    global  L, Constrain
    L = A
    Constrain = B
    
def getTransRecal(f,f0,f1,f2,f3,RB85frac,T,TDopp):
    p_dict = {'Bfield':0,'rb85frac':RB85frac,'Btheta':0,'lcell':L,\
          'T':T,'Dline':'D2','Elem':'Rb','Constrain':Constrain,'DoppTemp':TDopp}
    f_adjust = (f3*f**3 + f2*f**2 + f1*f - f0)*GHz/MHz # an attempt to account for the nonlinearity of scan
    # f_adjust= f
    y=get_spectra(f_adjust,[1,0,0],p_dict,outputs=['S0'])
    return np.asarray(np.real(y[0]))

# def getTrans(f,T):
#     p_dict = {'Bfield':0,'rb85frac':RB85frac,'Btheta':0,'lcell':L,\
#           'T':T,'Dline':'D2','Elem':'Rb','Constrain':True,'DoppTemp':T}
#     y = get_spectra(f*GHz/MHz,[1,0,0],p_dict,outputs=['S0'])
#     return np.asarray(np.real(y[0])) 

#%%
def extractTemp(f_array, data_signal, p0, plotFlag):
    indices = np.where(np.logical_and(f_array<60, f_array>-40))
    
    _bounds = ((-np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0), (np.inf, np.inf,np.inf, np.inf, 100, 200,200))
    popt, pcov = curve_fit(getTransRecal, f_array[indices], data_signal[indices], p0,bounds=_bounds)
    # f0,f1,f2,f3,RB85frac,T,TDopp
    _fit = getTransRecal(f_array,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])
    f_adjust = popt[3]*f_array**3  + popt[2]*f_array**2 + popt[1]*f_array - popt[0]
    
    residual = 100*(data_signal-_fit)
    tot_res = np.sum((residual/100)**2)
    res_max = max(np.abs(residual))
    
    RbFrac = round(popt[4],2)
    temp = round(popt[5],2)
    temp_Doppler = round(popt[6],2)
    # print(popt)
    fu=max(f_adjust)
    fl=min(f_adjust)
    if(plotFlag == 1):
        plt.close(1)
        fig3 = plt.figure(1)
        grid = plt.GridSpec(5,1,hspace=0.2)
        main_ax = fig3.add_subplot(grid[0:4])
        res_ax = fig3.add_subplot(grid[4])
        
        main_ax.plot(f_adjust,data_signal,'-',linewidth=2,color=cc('tab:blue',1),label='Measured signal')
        main_ax.plot(f_adjust,_fit,'--',linewidth=1.5,color=cc('tab:red',1),label='Fit')
        main_ax.set_xlim([fl,fu])
        main_ax.set_xticks([])
        main_ax.set_ylabel('Transmission')
        main_ax.set_title(r'Temp, Temp_Dopp = %.2f$^\mathrm{o}$C, %.2f$^\mathrm{o}$C' % (temp, temp_Doppler))
        main_ax.legend(loc="lower right")
        
        
        res_ax.plot(f_adjust,residual,color=cc('tab:orange',1),linewidth=1,label=r'%.2f' % tot_res)
        res_ax.fill_between(f_adjust,residual,color=cc('tab:orange',0.2))
        res_ax.legend(loc="lower left")
        res_ax.set_ylim(np.asarray([-res_max,res_max])*1.25)
        res_ax.set_xlim([fl,fu])
        res_ax.set_ylabel('Residuals [%]')
        res_ax.set_xlabel('Freq. [GHz]')
    else:
        pass
    

    
    
    return [f_adjust, tot_res, temp, temp_Doppler, RbFrac, popt], [data_signal,_fit,residual, res_max]
