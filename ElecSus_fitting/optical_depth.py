import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, '/home/otps3141/Documents/Dokumente/ETH QE/Master Thesis Imperial/Thesis/Code/OBEsimulation')

import AtomicConstants as AC


def get_OD(F1_weight, F2_weight, T, L, effective=False):
    """ Returns the optical depth for Rb87 with F1 and F2 weights in GS manifold """

    kB = 1.38e-23
    hbar = 1.05457182e-34
    eps0 = 8.8541878128e-12
    c = 299792458

    atom87_F1_config = {"Hyperfine splitting": True, "states": {"initial": {"n": 5, "J": 1/2, "L":0, "F": 1}, "intermediate": {"n": 5, "L":1, "J": 3/2}, 
                                                        "storage": {"n" : 4, "L": 2, "J" : 5/2}}}
    

    atom87_F2_config = {"Hyperfine splitting": True, "states": {"initial": {"n": 5, "J": 1/2, "L":0, "F": 2}, "intermediate": {"n": 5, "L":1, "J": 3/2}, 
                                                        "storage": {"n" : 4, "L": 2, "J" : 5/2}}}
    
    
    atom87_F1 = AC.Rb87(atom87_F1_config)
    atom87_F2 = AC.Rb87(atom87_F2_config)
    
    def pV(T):
        """ Vapour pressure of Cs as a function of temperature (K) """
        # liquid phase, T>25C
        return pow(10, -94.04826 - 1961.258/T - 0.03771687*T + 42.575 * np.log10(T))
    
    def Nv(T):
        """ Number density as a function of temperature (K) """
        # convert from torr to Pa
        return 133.323*pV(T)/(kB*T)
    
    def optical_depth(atom, T, L):
        cross_section = pow(atom.reduced_dipoles[0], 2)*atom.angular_frequencies[0]/(atom.decay_rates[0]*eps0*hbar*c)
        OD = Nv(T)*L*cross_section
        return OD
    
    def effective_optical_depth(atom, OD, T):
        width = np.sqrt(kB*T/(atom.mass*pow(c, 2)))*atom.angular_frequencies[0]
        ODdash = OD*atom.decay_rates[0]/(2*width) * np.sqrt(np.pi*np.log(2))
        return ODdash
    
    if effective == False:
        return F1_weight * optical_depth(atom87_F1, T, L) + F2_weight * optical_depth(atom87_F2, T, L)
    
    else:
        return F1_weight * effective_optical_depth(atom87_F1, optical_depth(atom87_F1, T, L), T) + F2_weight * effective_optical_depth(atom87_F2, optical_depth(atom87_F2, T, L), T)


    