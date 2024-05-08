import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, savgol_filter
import extractTemperature_auto as ET
cwd = os.getcwd()
import definitions as defs
from elecsus.libs import numberDensityEqs as n
from scipy.constants import physical_constants, c

folder = cwd[:-22]+'Data\\VapourTemperature\\2023-11-30-CellTemperature\\'

filename = '20231130_OD_136oC.csv'


CSVData = open(folder+filename).readlines()[:-1]
data = np.loadtxt(CSVData, delimiter=",",skiprows=4)
# data=np.loadtxt(open('/Users/epy/file2.txt','rt').readlines()[:-1], skiprows=1, dtype=None)

#%%

# Choose range of data that gives single scan
r1=550
r2=1500#len(data)


plt.figure(1,clear=True)
plt.plot(-data[r1:r2,0],data[r1:r2,1])
plt.plot(-data[r1:r2,0],data[r1:r2,2])


#%%

#smooth data

y_satspec=savgol_filter(data[r1:r2,1],window_length=11,polyorder=1)
y_cell=savgol_filter(data[r1:r2,2],window_length=11,polyorder=1)


plt.figure(1,clear=True)
# plt.plot(data[r1:r2,0],data[r1:r2,1])
plt.plot(-data[r1:r2,0],y_satspec)
# plt.plot(data[r1:r2,0],data[r1:r2,2])
plt.plot(-data[r1:r2,0],y_cell)

#%%

#normalise data
y_data_satspec = y_satspec/max(y_satspec)
y_data_cell = (y_cell-min(y_cell))/max((y_cell-min(y_cell)))
time_data=data[r1:r2,0]


#Position of peaks in GHz
_Rb87F1 = 4.15; _Rb85F1 = 1.675; _Rb85F2 = -1.232; _Rb87F2 = -2.366


OD = -np.log(y_data_satspec/max(y_data_satspec))
peak_points, _ = find_peaks(OD, prominence=0.05, distance=20)
print(peak_points)
plt.figure(2,clear=True)
plt.plot(OD)
plt.plot(peak_points,OD[peak_points],'x')

#%%
#Choose which order dips appear in from L to R (depends on direction of scan)
# If two closest dips are on left then choose direction =1:
# If two closest dips are on right then choose direction =-1:
    
direction =1

if direction ==1:
    f_points = np.asarray([_Rb87F1,_Rb85F1,_Rb85F2,_Rb87F2])
else:
    f_points = np.asarray([_Rb87F2,_Rb85F2,_Rb85F1,_Rb87F1])

f_array = np.polyval(np.polyfit(peak_points,f_points,1),np.asarray(range(0,len(time_data))))

#get indices of background data (away from dips) 
w=1 #width in GHz away from peaks to include in 'background'

if direction == 1:
    indices_spec = np.where(f_array<(f_points[3]-w))
    indices2 = np.where(np.logical_and(f_array<(f_points[2]-w), f_array>(f_points[3]+w)))
    indices3 = np.where(np.logical_and(f_array>(f_points[2]+w),f_array<(f_points[1] - w)))
    indices4 = np.where(np.logical_and(f_array>(f_points[0]+w),f_array>(4.5)))
    indices_spec = np.append(indices_spec, [indices2])
    indices_spec = np.append(indices_spec, [indices3])
    indices_spec = np.append(indices_spec, [indices4])
else:
    indices_spec = np.where(f_array<(f_points[0]-w))
    indices2 = np.where(np.logical_and(f_array<(f_points[1]-w), f_array>(f_points[0]+w)))
    indices3 = np.where(np.logical_and(f_array>(f_points[1]+w),f_array<(f_points[2] - w)))
    indices4 = np.where(np.logical_and(f_array>(f_points[3]+w),f_array>(4.5)))
    indices_spec = np.append(indices_spec, [indices2])
    indices_spec = np.append(indices_spec, [indices3])
    indices_spec = np.append(indices_spec, [indices4])

#fit polynomial to background data (away from dips) to account for slope
p = np.polyfit(f_array[indices_spec], y_data_satspec[indices_spec], 3)
bg = np.polyval(p, f_array)
y_data_satspec_bg = y_data_satspec/bg

y_data_cell_bg = 1*(y_data_cell/bg)/max(y_data_cell/bg)

#plot data
plt.figure(2)
plt.clf()
plt.cla()
plt.plot(f_array[indices_spec], y_data_satspec[indices_spec],'o') #indices used in background calculation
plt.plot(f_array,y_data_satspec_bg)
plt.plot(f_array,y_data_cell_bg)

#positions of Rb dips
plt.plot([_Rb87F1,_Rb87F1],[0,1],'k--')
plt.plot([_Rb85F1,_Rb85F1],[0,1],'k--')
plt.plot([_Rb85F2,_Rb85F2],[0,1],'k--')
plt.plot([_Rb87F2,_Rb87F2],[0,1],'k--')
plt.xlabel('Freq (GHz)')





#%%
#Set initial guess parameters
RbFrac = 3 # percentage of Rb 85 in cell
L = 0.08 # length of cell 
temp_init = 70
temp_Dopp = 50
Constrain = False # choose true to have the temp and Doppler temp constrained, false to have as independent parameters
p0=[0, 1, -1e-3,1e-6, temp_init, temp_Dopp, RbFrac]


ET.globalVars(L,Constrain) # send through the variables 
[freq, residuals, temperature, temp_Doppler, Rb85Frac, popt] , [data_signal,_fit,residual, res_max]= ET.extractTemp(f_array, y_data_cell_bg, p0, 1)


#%%
kB=physical_constants['Boltzmann constant'][0] 
amu=physical_constants['atomic mass constant'][0] #An atomic mass unit in kg

Gamma_D2 = 6.065e6
trans_dipole_D2 = 3.584e-29
Wav_D2 = 780.241209686e-9
f = 300e-3; w_i = 1e-3/2; waist = Wav_D2*f/(np.pi*w_i)
num_density = n.CalcNumberDensity(temperature+273.15,'Rb87')*(1-RbFrac/100)
vol = defs.getVol(waist,L,Wav_D2)
N = num_density*vol

OD = defs.OD(trans_dipole_D2,Wav_D2,N,waist,Gamma_D2)
mass = 86.909180520*amu
a = np.sqrt(kB*(temp_Doppler+273.15)/mass)
U = np.sqrt(2)*a


f0 =384230426.6e6
DoppBroad = f0*np.sqrt(8*kB*(temp_Doppler+273.15)*np.log(2)/(mass*c**2))

# U = np.sqrt(a**2*(3*np.pi - 8)/np.pi)
print('OD = ' + str(OD))
print('Number of Atoms = ' + str(np.round(N/1e9,1)) + ' x 10^9')
print('Dopp. Broad. = ' + str(np.round(1e-6*DoppBroad,2)) + ' MHz')



#%%

data =np.transpose([(f_array),(y_data_cell_bg)])
np.savetxt(filename,data,delimiter=',')

