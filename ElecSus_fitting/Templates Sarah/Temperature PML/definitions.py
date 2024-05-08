# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:27:17 2022

@author: pl3m20
"""
import numpy as np
def OD(dipole,wavelength,N,beam_radius,gamma):
    #  dipole in Cm
    #  wavelength in m
    #  beam radius in m
    #  gamma in Hz
    
    hbar = 1.0545718e-34;
    c = 299792458;
    eps0 = 8.854187817e-12;
    
    gamma2pi = 2*np.pi*gamma;
    omega_s = 2*np.pi*c/wavelength;
    A = np.pi*(beam_radius)**2;
    
    
    OD = (dipole/hbar)**2*hbar*omega_s*N/(2*eps0*c*A*gamma2pi);
    
    return OD
    # % for SiV: theDCalculator(4.7e-29,737e-9,100,450e-9,100e6)
    # % for Cs: theDCalculator(3e-29,852e-9,1e9,140e-6,6e6)
    # % for PrYSO: theDCalculator(4e-32,606e-9,1e15,140e-6,1e3)
    # % for PrYSO stoicmetric: theDCalculator(4e-32,606e-9,1e15,140e-6,1e3)
    

def getVol(waist,L,wavelength):
    zR = np.pi*waist**2/wavelength
    vol = np.pi*waist**2*L*(1 + (1/3)*(L/2/zR)**2)
    return vol
    
    