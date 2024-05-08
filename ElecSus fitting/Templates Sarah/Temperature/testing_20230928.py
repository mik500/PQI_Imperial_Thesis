import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import extractTemperature_auto as ET
cwd = os.getcwd()

folder = cwd[:-22]+'Data\\VapourTemperature\\2023-09-28-CellTemperature\\'

filename = 'od-cellTemp136.csv'


CSVData = open(folder+filename)
data = np.loadtxt(CSVData, delimiter=",",skiprows=2)

# plt.figure(1,clear=True)
# plt.plot(data[:,0],data[:,1])
# plt.plot(data[:,0],data[:,2])

#%%

# Choose range of data that gives single scan
r1=250
r2=1250



#normalise data
y_data_satspec = data[r1:r2,1]/max(data[r1:r2,1])
y_data_cell = (data[r1:r2,2]-min(data[r1:r2,2]))/max((data[r1:r2,2]-min(data[r1:r2,2])))
time_data=data[r1:r2,0]


#Position of peaks in GHz
_Rb87F1 = 4.15; _Rb85F1 = 1.675; _Rb85F2 = -1.232; _Rb87F2 = -2.366


OD = -np.log(y_data_satspec/max(y_data_satspec))
peak_points, _ = find_peaks(OD, prominence=0.02, distance=30)
print(peak_points)
plt.plot(OD)
plt.plot(peak_points,OD[peak_points],'x')

#Choose which order dips appear in from L to R (depends on direction of scan)
# f_points = np.asarray([_Rb87F1,_Rb85F1,_Rb85F2,_Rb87F2])
f_points = np.asarray([_Rb87F2,_Rb85F2,_Rb85F1,_Rb87F1])

f_array = np.polyval(np.polyfit(peak_points,f_points,1),np.asarray(range(0,len(time_data))))

#get indices of background data (away from dips) 
w=1 #width in GHz away from peaks to include in 'background'
indices_spec = np.where(f_array<(f_points[0]-w))
indices2 = np.where(np.logical_and(f_array<(f_points[1]-w), f_array>(f_points[0]+w)))
indices3 = np.where(np.logical_and(f_array>(f_points[1]+w),f_array<(f_points[2] - w)))
indices4 = np.where(np.logical_and(f_array>(f_points[3]+w),f_array>(4.5)))
indices_spec = np.append(indices_spec, [indices2])
indices_spec = np.append(indices_spec, [indices3])
indices_spec = np.append(indices_spec, [indices4])

#fit polynomial to background data (away from dips) to account for slope
p = np.polyfit(f_array[indices_spec], y_data_satspec[indices_spec], 2)
bg = np.polyval(p, f_array)
y_data_satspec_bg = y_data_satspec/bg

#get indices of background data (away from dips) for high temperature cell
w1=2.6 #using two widths here, firstly at edges of plot, and secondly in the centre
w2=1.2
indices_cell = np.where(f_array<(f_points[0]-w1))
indices2 = np.where(np.logical_and(f_array<(f_points[1]-w1), f_array>(f_points[0]+w1)))
indices3 = np.where(np.logical_and(f_array>(f_points[1]+w2),f_array<(f_points[2] - w2)))
indices4 = np.where(np.logical_and(f_array>(f_points[3]+w1),f_array>(4.5)))
indices_cell = np.append(indices_cell, [indices2])
indices_cell = np.append(indices_cell, [indices3])
indices_cell = np.append(indices_cell, [indices4])

p = np.polyfit(f_array[indices_cell], y_data_cell[indices_cell], 2)
bg = np.polyval(p, f_array)

y_data_cell_bg = (y_data_cell/bg)/max(y_data_cell/bg)

#plot data
plt.figure(2)
plt.clf()
plt.cla()
plt.plot(f_array[indices_spec], y_data_satspec[indices_spec],'o') #indices used in background calculation
plt.plot(f_array[indices_cell], y_data_cell[indices_cell],'o')#indices used in background calculation
plt.plot(f_array,y_data_satspec_bg)
plt.plot(f_array,y_data_cell_bg)

#positions of Rb dips
plt.plot([_Rb87F1,_Rb87F1],[0,1],'k--')
plt.plot([_Rb85F1,_Rb85F1],[0,1],'k--')
plt.plot([_Rb85F2,_Rb85F2],[0,1],'k--')
plt.plot([_Rb87F2,_Rb87F2],[0,1],'k--')
plt.xlabel('Freq (GHz)')


#Set initial guess parameters
RbFrac = 3 # percentage of Rb 85 in cell
L = 0.08 # length of cell 
temp_init = 70
temp_Dopp = 70
Constrain = False # choose true to have the temp and Doppler temp constrained, false to have as independent parameters
p0=[0, 1, -1e-3,1e-6, temp_init, temp_Dopp, RbFrac]


ET.globalVars(L,Constrain) # send through the variables 
[freq, residuals, temperature, temp_Doppler, Rb85Frac, popt] , [data_signal,_fit,residual, res_max]= ET.extractTemp(f_array, y_data_cell_bg, p0, 1)


