import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
from scipy.special import erfinv
import scipy.linalg as linalg
from sympy.physics.quantum.cg import Wigner6j
from sympy.physics.quantum.cg import CG
import timeit
import itertools

class solver:
    """
    Class to hold grids of coherences, insert boundary conditions, hold decays and couplings.

    """ 
    def __init__(self, atom, protocol, Einits, Sinits, deltas, OD, L, n, m, tbounds, T, vno):
        # Constants
        self.kB = 1.38e-23 
        self.M = atom.mass #1.443e-25
        self.c = 299792458
        self.hbar = 1.05457182e-34

        self.energy_tol = 1.01 # allow for 1% increase in energy of simualtion from numerical error
        
        self.n = n # space steps
        self.m = m # time steps
        self.tbounds = tbounds
        
        self.zCheby = self.cheby(np.linspace(0, n-1, n)) # zpoints
        
        self.T = T # atom temperature
        self.vno = vno # number of velocity classes
        self.OD = OD # stationary optical depth
        self.L = L # atomic cell length

        self.vs = self.vclasses() # create velocity classes
        self.dvs = self.dv_construct(self.vs) # velocity class widths

        self.Lsim = self.set_Lsim() # simulation length : have one point either side of cell for boudary conditions
        self.nz = self.set_nz() # atom number density as a function of z

        self.protocol = protocol # memory protocol identification string
        #self.config = config # atomic configuration identification string
        self.atom = atom

        self._deltaS = deltas[0] # signal detuning, not in natrual units
        if self.protocol == '4levelTORCAP' or self.protocol == '4levelTORCAG' or self.protocol == '4levelORCA' or self.protocol == 'TORCAP_2dressing_states': # for protocols involving two cotnrol fields, have second control field detuning
            self._deltaC = np.array([deltas[1], deltas[2]])
        elif self.protocol == 'TORCA_GSM_D1' or self.protocol == 'TORCA_GSM_D2' or self.protocol == 'ORCA_GSM':
            self._deltaC = np.array([deltas[1], deltas[2], deltas[3]]) # control field and two fields to map down to ground state
        else:
            self._deltaC = deltas[1] # control field detuning, not in natural units

        if self.protocol == 'EITFWM' or self.protocol == 'RamanFWM' or self.protocol == 'RamanFWM-Magic': # if have FWM, check if initial condition for FWM field
            if np.array(Einits).shape[0] != 2:
                Einits = np.array([Einits, np.zeros((m, 2))])

        self._Einits = Einits # electric field of photon initial condition into the cell as an array which is a funcion of time, not in natural units
        self.Sinits = Sinits # atomic coherences initial condition at start of simulation, an array as a function of z

        #Define photon grid
        self.E = np.zeros((self.m, self.n, 2), dtype=complex) #(t, z, polarisation = (L, R))

        self.coherences_config()

        self.Db = self.Db_constructor()
        #used to solve linear equation Db*E = P, solving for E
        #self.lu, self.piv = linalg.lu_factor(self.Db)
        # self.Db_inv = np.zeros((self.n, self.n, 2, 2), dtype=complex)
        # for i in range(2):
        #     for j in range(2):
        #         if self.Db[:, :, i, j].flatten()[1:].any():
        #             self.Db_inv[:, :, i, j] = np.linalg.inv(self.Db[:, :, i, j])
        #         else:
        #             self.Db_inv[:, :, i, j] = 0

        self.Db_inv = self.invert_D(self.Db)

        if self.protocol == 'EITFWM' or self.protocol == 'RamanFWM' or self.protocol == 'RamanFWM-Magic': # if have FWM, need another differential matrix to account for different dispersion of FWM field 
            #Define FWM photon grid
            self.EF = np.zeros((self.m, self.n, 2), dtype=complex) #(t, z, polarisation = (L, R))
            self.DbF = self.DbF_constructor()
            self.DbF_inv = self.invert_D(self.DbF)

            # self.DbF_inv = np.zeros((self.n, self.n, 2, 2), dtype=complex)
            # for i in range(2):
            #     for j in range(2):
            #         if self.DbF[:, :, i, j].flatten()[1:].any():
            #             self.DbF_inv[:, :, i, j] = np.linalg.inv(self.DbF[:, :, i, j])
            #         else:
            #             self.DbF_inv[:, :, i, j] = 0

        if self.protocol == 'ORCA_GSM':
            # Define resonant photon field
            self.ER = np.zeros((self.m, self.n, 2), dtype=complex) #(t, z, polarisation = (L, R))


        self.solved = False

    def invert_D(self, D):
        D_inv = np.zeros((self.n, self.n, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                if D[:, :, i, j].flatten()[1:].any():
                    D_inv[:, :, i, j] = np.linalg.inv(D[:, :, i, j])
                else:
                    D_inv[:, :, i, j] = 0

        return D_inv


    def metadata(self):        
        metadata = {
                    "protocol": f"{self.protocol}",
                    "config": f"{self.atom}",
                    "deltas": self._format_deltas(self._deltaS, self._deltaC),
                    "OD": f"{str(round(self.OD)).replace('.', 'd')}",
                    "L": f"{str(self.L).replace('.', 'd')}",
                    "n": f"{self.n}",
                    "m": f"{self.m}",
                    "tbounds": f"{str(self.tbounds).replace('.', 'd')}",
                    "T": f"{str(self.T).replace('.', 'd')}",
                    "vno": f"{self.vno}"
                }
        return metadata
    
    def _format_deltas(self, deltaS, deltaC):
        if isinstance(deltaC, np.ndarray):
            return f"[{deltaS*1e-6/(2*np.pi)}, {(deltaC*1e-6/(2*np.pi)).tolist()}]"
        return f"[{deltaS*1e-6/(2*np.pi)}MHz, {deltaC*1e-6/(2*np.pi)}MHz]".replace('.', 'd')
    
    def __repr__(self):
        return str(self.metadata())

    def cheby(self, i):
        """
        Generates Chebyshev points in the cell.
        """
        zrange = 1
        center = 0.5
        zj = center- (zrange/2)*np.cos( (np.pi*i/(self.n-1)) )
        return zj
    
    def Db_constructor(self):
        """
        Constructs Chebyshev differentiation matrix including boundary condition of E
        """
        ci = np.ones(self.n)
        ci[0] *= 2
        ci[-1] *= 2
        ci *= pow(-1, np.linspace(0, self.n-1, self.n))
        Z = np.tile(self.zCheby, (self.n, 1))
        dZ = Z.T-Z
        D = ( np.outer(ci,(1/ci)) )/(dZ + np.eye(self.n, dtype=complex)) #off diagonal
        D = D - np.diag(np.sum(D.T, axis=0)) 

        #make into (n, n, Q=2, Q) shape
        D = np.transpose( np.tile(D, (2, 2, 1, 1)), (2, 3, 0, 1) )
        D[:, :, 0, 1] = 0
        D[:, :, 1, 0] = 0

        if self.protocol == 'ORCA' or self.protocol == '4levelORCA' or self.protocol == 'Raman' or self.protocol == 'RamanFWM' or self.protocol == 'RamanFWM-Magic': 
            # dsqrtQ = (z, g, mg, j, mj, v, Q)
            # DELTAS = (g, j, v)
            # sum over (g, mg, j, mj, v)
            # D[:, :, :, 0] += np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (self.dsqrtQ[..., 0])[..., None])/(1 + 1j*self.DELTAS[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            # D[:, :, :, 1] += np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (self.dsqrtQ[..., 1])[..., None])/(1 + 1j*self.DELTAS[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]

            #D += np.einsum('NzQp, Nz -> NzQp' , np.einsum('NghjkvQ, zghjkvp -> NzQp', self.dsqrtQ, np.einsum('zghjkvp, gjv -> zghjkvp', self.dsqrtQ, 1/(1+1j*self.DELTAS))), np.eye(self.n))
            D += np.einsum('zQp, Nz -> NzQp' , np.einsum('zghjkvQ, zghjkvp -> zQp', self.dsqrtQ, np.einsum('zghjkvp, gjv -> zghjkvp', self.dsqrtQ, 1/(1+1j*self.DELTAS))), np.eye(self.n))
            #D += np.einsum('NzQp, Nz -> NzpQ' , np.einsum('NghjkvQ, zghjkvp -> NzQp', self.dsqrtQ, np.einsum('zghjkvp, gjv -> zghjkvp', self.dsqrtQ, 1/(1+1j*self.DELTAS))), np.eye(self.n))




        #boundary conditions
        D[0] *= 0 
        D[0, 0, :, :] = 1
        return D 
    
    def DbF_constructor(self):
        """
        Constructs Chebyshev differentiation matrix including boundary condition of E
        """
        ci = np.ones(self.n)
        ci[0] *= 2
        ci[-1] *= 2
        ci *= pow(-1, np.linspace(0, self.n-1, self.n))
        Z = np.tile(self.zCheby, (self.n, 1))
        dZ = Z.T-Z
        D = ( np.outer(ci,(1/ci)) )/(dZ + np.eye(self.n, dtype=complex)) #off diagonal
        D = D - np.diag(np.sum(D.T, axis=0)) 

        #make into (n, n, Q=2, Q) shape
        D = np.transpose( np.tile(D, (2, 2, 1, 1)), (2, 3, 0, 1) )
        D[:, :, 0, 1] = 0
        D[:, :, 1, 0] = 0

        if self.protocol == 'EITFWM' or self.protocol == 'RamanFWM': 
            # dsqrtQ = (z, g, mg, j, mj, v, Q)
            # DELTAS = (g, j, v)
            # sum over (g, mg, j, mj, v)
            D[:, :, :, 0] += np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (self.dsqrtQ[..., 0])[..., None])/(1 + 1j*self.DELTASHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            D[:, :, :, 1] += np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (self.dsqrtQ[..., 1])[..., None])/(1 + 1j*self.DELTASHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
        #boundary conditions
        D[0] *= 0
        D[0, 0, :, :] = 1
        return D 
    
    def MB(self, v):
        if (self.T == 0):
            return 1
        else:
            sigma = np.sqrt((self.kB*self.T)/self.M)
            return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-pow(v,2)/(2*pow(sigma,2)))
        
    def vclasses(self):
        """
        Generates velocity classes 
        """        
        sigma = np.sqrt((self.kB*self.T)/self.M)
        x = np.linspace(-0.99, 0.99, self.vno)
        y = np.array([erfinv( i ) for i in x])
        #vrange = -3.5 sigma to +3.5 sigma
        vs = 3.5*np.sqrt((self.kB*self.T)/self.M)*y/max(y)
        return vs
    
    def dv_construct(self, vs):
        if len(vs) == 1:
            dvfull = np.array([1.0])
        else:
            zero_and_positive = np.concatenate([[0.0], vs[int(len(vs)/2):]])
            du = np.diff(zero_and_positive)
            dvhalf = []
            for i in range(0, len(du)):
                dvhalf.append(du[i])
                if i>0:
                    dvhalf[i] -= dvhalf[i-1]

            dv = np.array(dvhalf) * 2

            dvfull = np.concatenate([np.flip(dv), dv])
        return dvfull
    
    def set_Lsim(self):
        """
        This is the length used in the simulation.  Equal to the length of atomic cell plus two end Chebyshev points.
        """
        Lsim = np.abs(self.zCheby[-1] - self.zCheby[0])/np.abs(self.zCheby[-2] - self.zCheby[1]) * self.L
        return Lsim
    
    def set_nz(self):
        """
        Set normalised number density of atoms to be zero outside of cell and one inside.
        """
        nz = np.ones((self.n,))
        nz[0] = 0
        nz[-1] = 0
        return nz

    def coherences_config(self):
        self.atom_constants_unpack()
        self.deltaS = self._deltaS/self.gamma
        self.deltaC = self._deltaC/self.gamma
        self.cNU = self.c/(self.L*self.gamma)
        self.tpoints = np.linspace(self.tbounds[0]*self.gamma,self.tbounds[1]*self.gamma,self.m)
        self.tstep = self.tpoints[1] - self.tpoints[0] 
        self.t_grid, self.z_grid = np.meshgrid(self.tpoints, self.zCheby)
        
        self.detunings(self.deltaS, self.deltaC)
        self.calc_population()
        if self.protocol == 'TORCA' or self.protocol == 'ORCA' or self.protocol == 'Raman' or self.protocol == 'Raman_test':
            self.Einits = interp1d(self.tpoints, self._Einits/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.S = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.S[0, :, ..., :] = self.Sinits
            self.coherences_list = [self.S]
        elif self.protocol == 'TORCAP':
            self.Einits = interp1d(self.tpoints, self._Einits/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.Pge = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex)
            self.Pes = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex)
            self.S = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.Yee = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex) # coherence between hyperfine intermediate states due to strong control field on bottom transition
            self.Mgg = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fg), len(self.atom.mg), self.vno), dtype=complex) # coherence between zeeman sublevels of ground state due to strong control field on bottom transition
            self.Ycorrection = self.coherence_corr(self.Yee)
            self.Mcorrection = self.coherence_corr(self.Mgg)
            self.S[0, :, ..., :] = self.Sinits
            self.coherences_list = [self.Pge, self.Pes, self.S, self.Yee, self.Mgg]
        elif self.protocol == 'RamanFWM':
            self.Einits = interp1d(self.tpoints, self._Einits[0]/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.EFinits = interp1d(self.tpoints, self._Einits[1]/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial FWM photon condition, in natural units
            self.S = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.Yee = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex) # coherence between hyperfine intermediate states due to strong control field on bottom transition
            self.Mgg = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fg), len(self.atom.mg), self.vno), dtype=complex) # coherence between zeeman sublevels of ground state due to strong control field on bottom transition
            self.Ycorrection = self.coherence_corr(self.Yee)
            self.Mcorrection = self.coherence_corr(self.Mgg)
            self.S[0, :, ..., :] = self.Sinits
            self.coherences_list = [self.S, self.Yee, self.Mgg]
        elif self.protocol == 'EIT' or self.protocol == 'EITFWM' or self.protocol == 'FLAME':
            self.Einits = interp1d(self.tpoints, self._Einits/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.P = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of excited states, number of mF states in excited state with largest F, number of velocity classes)
            self.S = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.S[0, :, ..., :] = self.Sinits
            self.coherences_list = [self.P, self.S]
        elif self.protocol == 'RamanFWM-Magic':
            self.Einits = interp1d(self.tpoints, self._Einits[0]/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.EFinits = interp1d(self.tpoints, self._Einits[1]/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial FWM photon condition, in natural units
            self.PF = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of excited states, number of mF states in excited state with largest F, number of velocity classes)
            self.S = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.Yee = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex) # coherence between hyperfine intermediate states due to strong control field on bottom transition
            self.Mgg = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fg), len(self.atom.mg), self.vno), dtype=complex) # coherence between zeeman sublevels of ground state due to strong control field on bottom transition
            self.Ycorrection = self.coherence_corr(self.Yee)
            self.Mcorrection = self.coherence_corr(self.Mgg)
            self.S[0, :, ..., :] = self.Sinits
            self.coherences_list = [self.PF, self.S, self.Yee, self.Mgg]
        elif self.protocol == '4levelTORCAP':
            self.Einits = interp1d(self.tpoints, self._Einits[0]/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.Pge = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex)
            self.Pes = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex)
            self.Peb = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fb), len(self.atom.mb), self.vno), dtype=complex)
            self.Sgs = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.Sgb = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fb), len(self.atom.mb), self.vno), dtype=complex)
            self.Yee = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex) # coherence between hyperfine intermediate states due to strong control field on bottom transition
            self.Mgg = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fg), len(self.atom.mg), self.vno), dtype=complex) # coherence between zeeman sublevels of ground state due to strong control field on bottom transition
            self.Ycorrection = self.coherence_corr(self.Yee)
            self.Mcorrection = self.coherence_corr(self.Mgg)
            self.Sgs[0, :, ..., :] = self.Sinits[0] 
            self.Sgb[0, :, ..., :] = self.Sinits[1]
            self.coherences_list = [self.Pge, self.Pes, self.Peb, self.Sgs, self.Sgb, self.Yee, self.Mgg]
        elif self.protocol == 'TORCAP_2dressing_states':
            self.Einits = interp1d(self.tpoints, self._Einits[0]/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.Pge = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex)
            self.Pes = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex)
            self.Peb = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fb), len(self.atom.mb), self.vno), dtype=complex)
            self.Peb2 = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fb2), len(self.atom.mb2), self.vno), dtype=complex)
            self.Sgs = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.Sgb = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fb), len(self.atom.mb), self.vno), dtype=complex)
            self.Sgb2 = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fb2), len(self.atom.mb2), self.vno), dtype=complex)
            self.Yee = np.zeros((self.m, self.n, len(self.atom.Fj), len(self.atom.mj), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex) # coherence between hyperfine intermediate states due to strong control field on bottom transition
            self.Mgg = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fg), len(self.atom.mg), self.vno), dtype=complex) # coherence between zeeman sublevels of ground state due to strong control field on bottom transition
            self.Sgs[0, :, ..., :] = self.Sinits[0] 
            self.Sgb[0, :, ..., :] = self.Sinits[1]
            self.Sgb2[0, :, ..., :] = self.Sinits[2]
            self.coherences_list = [self.Pge, self.Pes, self.Peb, self.Peb2, self.Sgs, self.Sgb, self.Sgb2, self.Yee, self.Mgg]
        elif self.protocol == 'TORCA_GSM_D1' or self.protocol == 'TORCA_GSM_D2':
            self.Einits = interp1d(self.tpoints, self._Einits[0]/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.Pge2 = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fb), len(self.atom.mb), self.vno), dtype=complex)
            self.Sgs = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.Sgs2 = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fh), len(self.atom.mh), self.vno), dtype=complex)
            self.Sgs[0, :, ..., :] = self.Sinits[0] 
            self.Sgs2[0, :, ..., :] = self.Sinits[1]
            self.coherences_list = [self.Pge, self.Pge2, self.Sgs, self.Sgs2]
        elif self.protocol == 'ORCA_GSM':
            self.Einits = interp1d(self.tpoints, self._Einits[0]/np.sqrt(self.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            self.Pge = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex)
            self.Pge2 = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fj), len(self.atom.mj), self.vno), dtype=complex)
            self.Sgs = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fq), len(self.atom.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.Sgb = np.zeros((self.m, self.n, len(self.atom.Fg), len(self.atom.mg), len(self.atom.Fb), len(self.atom.mb), self.vno), dtype=complex)
            self.Sgs[0] = self.Sinits[0] 
            self.Sgb[0] = self.Sinits[1]
            self.coherences_list = [self.Pge, self.Pge2, self.Sgs, self.Sgb]
        # elif self.protocol == '4levelORCA':
        #     self.Sgs = np.zeros((self.m, self.n, len(self.Fg), len(self.mg), len(self.Fq), len(self.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
        #     self.Sgb = np.zeros((self.m, self.n, len(self.Fg), len(self.mg), len(self.Fb), len(self.mb), self.vno), dtype=complex)
        #     self.Sgs[0, :, ..., :] = self.Atomicinits[0]
        #     self.Sgb[0, :, ..., :] = self.Atomicinits[1]

    
    
    def detunings(self, deltaS, deltaC):
        self.detunings1(deltaS, deltaC)
        self.detunings2()

    def detunings1(self, deltaS, deltaC):
        if self.protocol == 'TORCA' or self.protocol == 'TORCAP':
            self.DELTAS = deltaS - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :]
            self.DELTAC = deltaC - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :]
            self.chi = np.tile(self.wj, (len(self.wj), 1)) - np.tile(self.wj, (len(self.wj), 1)).T # differences between statews in intermediate manifold
        elif self.protocol == '4levelTORCAP':
            self.DELTAS = deltaS - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :] #(j, q, v)
            self.DELTAC = deltaC[0] - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :] #(g, j, v)
            self.DELTAC2 = deltaC[1] - ( (self.wb[None, :] - self.wq[:, None]) )[:, :, None] + ((self.kcontrol2 * self.vs)/self.gamma)[None, None, :] #(q, b, v)
            self.chi = np.tile(self.wj, (len(self.wj), 1)) - np.tile(self.wj, (len(self.wj), 1)).T # differences between statews in intermediate manifold
        elif self.protocol == 'TORCAP_2dressing_states':
            self.DELTAS = deltaS - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :] #(j, q, v)
            self.DELTAC = deltaC[0] - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :] #(g, j, v)
            self.DELTAC2 = deltaC[1] - ( (self.wb[None, :] - self.wq[:, None]) )[:, :, None] + ((self.kcontrol2 * self.vs)/self.gamma)[None, None, :] #(q, b, v)
            self.DELTAC3 = deltaC[1] + self.dress_state_splitting - ( (self.wb2[None, :] - self.wq[:, None]) )[:, :, None] + ((self.kcontrol2 * self.vs)/self.gamma)[None, None, :] #(q, b, v)
            self.chi = np.tile(self.wj, (len(self.wj), 1)) - np.tile(self.wj, (len(self.wj), 1)).T # differences between statews in intermediate manifold
        elif self.protocol == 'TORCA_GSM_D1' or self.protocol == 'TORCA_GSM_D2':
            self.DELTAS = deltaS - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :] #(j, q, v)
            self.DELTAC = deltaC[0] + ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :] #(g, j, v)
            self.DELTAC2 = deltaC[1] + ( (self.wb[None, :] - self.wq[:, None]) )[:, :, None] + ((self.kcontrol2 * self.vs)/self.gamma)[None, None, :] #(q, b, v)
            self.DELTAC3 = deltaC[2] + ( (self.wa[None, :] - self.wb[:, None]) )[:, :, None] + ((self.kcontrol3 * self.vs)/self.gamma)[None, None, :] #(b, a, v)
        elif self.protocol == 'ORCA' or self.protocol == 'FLAME':
            self.DELTAS = deltaS - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :] #(j, q, v)
            self.DELTAC = deltaC - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :] #(g, j, v)
        elif self.protocol == '4levelORCA':
            self.DELTAS = deltaS - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :] #(j, q, v)
            self.DELTAC = deltaC[0] - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :] #(g, j, v)
            self.DELTAC2 = deltaC[1] - ( (self.wb[None, :] - self.wq[:, None]) )[:, :, None] + ((self.kcontrol2 * self.vs)/self.gamma)[None, None, :] #(q, b, v)
        elif self.protocol == 'ORCA_GSM':
            self.DELTAS = deltaS - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :] #(g, j, v)
            self.DELTAC = deltaC[0] - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :] #(j, q, v)
            self.DELTAC2 = deltaC[1] - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.kcontrol2 * self.vs)/self.gamma)[None, None, :] #(j, q, v)
            self.DELTAC3 = deltaC[2] - ( (self.wj[None, :] - self.wb[:, None]) )[:, :, None] + ((self.kcontrol3 * self.vs)/self.gamma)[None, None, :] #(j, q, v)
        elif self.protocol == 'RamanFWM' or  self.protocol == 'RamanFWM-Magic' or self.protocol == 'EITFWM':
            self.DELTAS = deltaS - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :]
            self.DELTAC = deltaC - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :]
            self.DELTACHF = deltaC - self.deltaHF - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :]
            self.DELTASHF = deltaC - 2*self.deltaHF - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :]
            self.chi = np.tile(self.wj, (len(self.wj), 1)) - np.tile(self.wj, (len(self.wj), 1)).T # differences between statews in intermediate manifold
        elif self.protocol == 'Raman' or self.protocol == 'Raman_test' or self.protocol == 'EIT':
            self.DELTAS = deltaS - ( (self.wj[None, :] - self.wg[:, None]) )[:, :, None] + ((self.ksignal * self.vs)/self.gamma)[None, None, :]
            self.DELTAC = deltaC - ( (self.wq[None, :] - self.wj[:, None]) )[:, :, None] + ((self.kcontrol * self.vs)/self.gamma)[None, None, :]

    def detunings2(self):
        if self.protocol == 'ORCA' or self.protocol == 'FLAME':
            self.DELTA2 = (self.DELTAS[..., None, :] + self.DELTAC[None, :, :, :])[:, 0, ...]
        elif self.protocol == 'TORCA' or self.protocol == 'TORCAP':
            self.DELTA2 = (self.DELTAS[None, ...] + self.DELTAC[:, :, None, :])[:, 0, ...]
        elif self.protocol == '4levelTORCAP':
            self.DELTAGS = (self.DELTAS[None, ...] + self.DELTAC[:, :, None, :])[:, 0, ...] #(g, q, v)
            self.DELTAGB = (self.DELTAGS[:, :, None, :] + self.DELTAC2[:, None, :, :])[:, 0, ...] #(g, b, v)
            self.DELTAEB = (self.DELTAS[:, :, None, :] + self.DELTAC2[None, :, :, :])[:, 0, ...] #(j, b, v)
        elif self.protocol == 'TORCAP_2dressing_states':
            self.DELTAGS = (self.DELTAS[None, ...] + self.DELTAC[:, :, None, :])[:, 0, ...] #(g, q, v)
            self.DELTAGB = (self.DELTAGS[:, :, None, :] + self.DELTAC2[:, None, :, :])[:, 0, ...] #(g, b, v)
            self.DELTAEB = (self.DELTAS[:, :, None, :] + self.DELTAC2[None, :, :, :])[:, 0, ...] #(j, b, v)
            self.DELTAGB2 = (self.DELTAGS[:, :, None, :] + self.DELTAC3[:, None, :, :])[:, 0, ...] #(g, b, v)
            self.DELTAEB2 = (self.DELTAS[:, :, None, :] + self.DELTAC3[None, :, :, :])[:, 0, ...] #(j, b, v)
        elif self.protocol == 'TORCA_GSM_D1' or self.protocol == 'TORCA_GSM_D2':
            self.DELTAGS = (self.DELTAS[None, ...] + self.DELTAC[:, :, None, :])[:, 0, ...] #(g, q, v)
            self.DELTAGB = (self.DELTAGS[:, :, None, :] + self.DELTAC2[:, None, :, :])[:, 0, ...] #(g, b, v)
            self.DELTAGS2 = (self.DELTAGB[:, :, None, :] + self.DELTAC3[None, :, :, :])[:, 0, ...] #(g, a, v)
        elif self.protocol == '4levelORCA':
            self.DELTAGS = (self.DELTAS[..., None, :] + self.DELTAC[None, :, :, :])[:, 0, ...] #(g, q, v)
            self.DELTAGB = (self.DELTAGS[:, :, None, :] + self.DELTAC2[:, None, :, :])[:, 0, ...] #(g, b, v)
        elif self.protocol == 'ORCA_GSM':
            self.DELTAGS = (self.DELTAS[..., None, :] + self.DELTAC[None, :, :, :])[:, 0, ...] #(g, q, v)
            #self.DELTAGS2 = (self.DELTAC2[None, :, :, :] + self.DELTAC3[:, :, None, :])[:, 0, ...] #(b, q, v)
            self.DELTAGJ = (self.DELTAGS[:, None, :, :] - self.DELTAC2[None, :, :, :])[:, :, 0, ...] #(g, j, v)
            self.DELTAGB = (self.DELTAGJ[:, :, None, :] - np.transpose(self.DELTAC3, (1, 0, 2))[None, :, :, :])[:, 0, :, ...] #(g, b, v)
            #self.DELTAGB = (self.DELTAGS[:, :, None, :] - np.transpose(self.DELTAGS2, (1, 0, 2))[None, :, :, :])[:, 0, :, ...] #(g, b, v)
        else:
            self.DELTA2 = (self.DELTAS[..., None, :] - self.DELTAC[None, :, :, :])[:, 0, ...]

    def calc_population(self):
        if self.T == 0:
            if len(self.atom.mg)>1: # assume population pumped to maximally stretched spin state, mg = +Fg, for the highest Fg
                self.pop = np.zeros((len(self.atom.Fg), len(self.atom.mg)))
                self.pop[-1, -1] = 1
            else:
                self.pop = np.ones((len(self.atom.Fg), len(self.atom.mg)))
        else:
            Eground = self.gamma*self.atom.wg/(2*np.pi) * self.hbar
            self.pop = np.repeat( np.exp(-Eground/(self.kB*self.T)), len(self.atom.mg)).reshape((len(self.atom.Fg), len(self.atom.mg))) #thermal distribution of populations
            if self.atom.Fg.ndim > 1:
                self.pop[ np.where( (len(self.atom.mg)-1)/2>self.atom.Fg  ), [0, -1]] = 0 #correct ground state population where mF level shouldn't exist
            self.pop = self.pop/sum(self.pop.flatten()) #normalise
        return None
    
    def atom_constants_unpack(self):
        self.gamma = self.atom.gammas[0]
        self.gammaS = self.atom.gammas[1]
        self.gammaSNU = self.gammaS/self.gamma
        self.wg = 2*np.pi*self.atom.wg/self.gamma
        self.wj = 2*np.pi*self.atom.wj/self.gamma
        self.wq = 2*np.pi*self.atom.wq/self.gamma
        if self.protocol == 'TORCA' or self.protocol =='TORCAP':
            self.dsqrtQ = self.atom.coupling_es
            self.OmegaQ = self.atom.coupling_ge
            self.kcontrol = self.atom.angular_frequencies[0]/self.c
            self.ksignal = self.atom.angular_frequencies[1]/self.c
        elif self.protocol == '4levelTORCAP':
            self.gammaB = self.atom.gammas[2]
            self.gammaBNU = self.gammaB/self.gamma
            self.dsqrtQ = self.atom.coupling_es
            self.OmegaQ = self.atom.coupling_ge
            self.OmegaQ2 = self.atom.coupling_sb
            self.kcontrol = self.atom.angular_frequencies[0]/self.c
            self.ksignal = self.atom.angular_frequencies[1]/self.c
            self.kcontrol2 = self.atom.angular_frequencies[2]/self.c
            self.wb = 2*np.pi*self.atom.wb/self.gamma
        elif self.protocol == 'TORCAP_2dressing_states':
            self.gammaB = self.atom.gammas[2]
            self.gammaBNU = self.gammaB/self.gamma
            self.gammaB2 = self.atom.gammas[3]
            self.gammaBNU2 = self.gammaB2/self.gamma
            self.dsqrtQ = self.atom.coupling_es
            self.OmegaQ = self.atom.coupling_ge
            self.OmegaQ2 = self.atom.coupling_sb
            self.OmegaQ3 = self.atom.coupling_sb2
            self.kcontrol = self.atom.angular_frequencies[0]/self.c
            self.ksignal = self.atom.angular_frequencies[1]/self.c
            self.kcontrol2 = self.atom.angular_frequencies[2]/self.c
            self.wb = 2*np.pi*self.atom.wb/self.gamma
            self.wb2 = 2*np.pi*self.atom.wb2/self.gamma
            self.rabi_modification = self.atom.reduced_dipoles[3]/self.atom.reduced_dipoles[2] # for rabi freq defined on sb transtion, converts to sb2 transition
            self.dress_state_splitting = 2*np.pi*self.atom.dress_state_splitting/self.gamma
        elif self.protocol == 'ORCA' or self.protocol == 'FLAME' or self.protocol == 'Raman' or self.protocol == 'Raman_test' or self.protocol == 'EIT':
            self.dsqrtQ = self.atom.coupling_ge
            self.OmegaQ = self.atom.coupling_es
            self.kcontrol = self.atom.angular_frequencies[1]/self.c
            self.ksignal = self.atom.angular_frequencies[0]/self.c
        elif self.protocol == '4levelORCA':
            self.gammaB = self.atom.gammas[2]
            self.gammaBNU = self.gammaB/self.gamma
            self.dsqrtQ = self.atom.coupling_ge
            self.OmegaQ = self.atom.coupling_es
            self.OmegaQ2 = self.atom.coupling_sd
            self.kcontrol = self.atom.angular_frequencies[1]/self.c
            self.ksignal = self.atom.angular_frequencies[0]/self.c
            self.kcontrol2 = self.atom.angular_frequencies[2]/self.c
            self.wb = 2*np.pi*self.atom.wb/self.gamma
        elif self.protocol == 'RamanFWM' or  self.protocol == 'RamanFWM-Magic' or self.protocol == 'EITFWM':
            self.dsqrtQ = self.atom.coupling_ge
            self.OmegaQ = self.atom.coupling_es

            self.dsqrtQF =  self.atom.coupling_es
            self.dsqrtQF = np.einsum('jkqwp, v -> jkqwvp', self.dsqrtQF, np.sqrt(self.MB(self.vs)*self.dvs)) # include velocity class distribution
            self.dsqrtQF *= np.sqrt(self.OD)
            self.dsqrtQF = np.einsum('jkqwvp, z -> zjkqwvp', self.dsqrtQF, self.nz) # account for outside of the cell, and potentially variable atomic density

            self.OmegaQF = self.atom.coupling_ge
            self.kcontrol = self.atom.angular_frequencies[1]/self.c
            self.ksignal = self.atom.angular_frequencies[0]/self.c
            self.deltaHF = 2*np.pi*self.atom.deltaHF/self.gamma
        elif self.protocol == 'ORCA_GSM':
            self.gammaB = self.atom.gammas[-1]
            self.gammaBNU = self.gammaB/self.gamma
            self.dsqrtQ = self.atom.coupling_ge # used for E and ER
            self.OmegaQ = self.atom.coupling_es # for control field and mapping field 1
            self.OmegaM2Q = self.atom.coupling_be # for mapping field 2
            self.ksignal = self.atom.angular_frequencies[0]/self.c
            self.kcontrol = self.atom.angular_frequencies[1]/self.c
            self.kcontrol2 = self.atom.angular_frequencies[2]/self.c
            self.kcontrol3 = self.atom.angular_frequencies[3]/self.c
            self.wb = 2*np.pi*self.atom.wb/self.gamma
            

        # photon couplings
        self.dsqrtQ = np.einsum('jkqwp, v -> jkqwvp', self.dsqrtQ, np.sqrt(self.MB(self.vs)*self.dvs)) # include velocity class distribution
        self.dsqrtQ *= np.sqrt(self.OD)
        self.dsqrtQ = np.einsum('jkqwvp, z -> zjkqwvp', self.dsqrtQ, self.nz) # account for outside of the cell, and potentially variable atomic density
      
    def co_prop(self, Control, t=np.array([])):
        """
        Control should be shape (t, p)
        """
        if t.any():
            _t, _z = np.meshgrid(t*self.gamma, self.zCheby) # makes grid for both t and z, using t function argument in natural units (rather than points to be used in simulation)
        else:
            # use simulation time axis
            _t, _z = np.meshgrid(self.tpoints, self.zCheby)
        Control3d_array = np.array([np.tile(Control[:, 0], self.n).reshape(self.n, len(Control[:, 0])), np.tile(Control[:, 1], self.n).reshape(self.n, len(Control[:, 1]))]).transpose(2, 1, 0)
        Control3d_func = LinearNDInterpolator(np.hstack([_t.reshape(-1, 1), _z.reshape(-1, 1)]), Control3d_array.transpose(1, 0, 2).reshape(-1, 2)/self.gamma, fill_value=0.0)
        return(Control3d_func) # f(t, z, p)
    
    def counter_prop(self, Control, t=np.array([]), zdef=0.5, field=0):
        """
        Control should be shape (t, p)
        zdef: what z position Control defined at
        field: used when multiple control fields are present
        """
        if t.any():
            t_NU = t*self.gamma # t in natural units
        else:
            t_NU = self.tpoints # t in natural units
            
        _t, _z = np.meshgrid(t_NU, self.zCheby) # makes grid for both t and z, using t function argument in natural units (rather than points to be used in simulation)
        if field == 0:
            self.DELTAC -= ((2*self.kcontrol * self.vs)/self.gamma)[None, None, :]
            self.detunings2()
        elif field==1:
            self.DELTAC2 -= ((2*self.kcontrol2 * self.vs)/self.gamma)[None, None, :]
            self.detunings2()
        elif field==2:
            self.DELTAC3 -= ((2*self.kcontrol3 * self.vs)/self.gamma)[None, None, :]
            self.detunings2()

        Control3d_array = []
        for zi in self.zCheby:
            Control3d_array.append([np.interp(t_NU+2*(zi-zdef)/self.cNU, t_NU, Control[:, 0]), np.interp(t_NU+2*(zi-zdef)/self.cNU, t_NU, Control[:, 1])])

        Control3d_array = np.array(Control3d_array).transpose(2, 0, 1)
        Control3d_func = LinearNDInterpolator(np.hstack([_t.reshape(-1, 1), _z.reshape(-1, 1)]), Control3d_array.transpose(1, 0, 2).reshape(-1, 2)/self.gamma, fill_value=0.0)
        return(Control3d_func) # f(t, z, p)
    
    def chebyshev_solver(self, p, Y=0, G=0):  
        if self.atom.splitting == True and (self.protocol == 'RamanFWM' or self.protocol == 'RamanFWM-Magic'):
            # need to change Db at each time step
            term1 = -np.einsum('zjkbxv, zghbxvp -> zghjkvp', Y, self.dsqrtQ)
            term2 = np.einsum('zghasv, zasbxvp -> zghjkvp', G, self.dsqrtQ)
            Db = np.zeros(self.Db.shape)
            Db[:, :, :, 0] = self.Db[:, :, :, 0] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (term1[..., 0])[..., None])/(1 + 1j*self.DELTAS[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            Db[:, :, :, 1] = self.Db[:, :, :, 1] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (term1[..., 0])[..., None])/(1 + 1j*self.DELTAS[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            Db[:, :, :, 0] = Db[:, :, :, 0] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (term2[..., 0])[..., None])/(1 + 1j*self.DELTAS[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            Db[:, :, :, 1] = Db[:, :, :, 1] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (term2[..., 0])[..., None])/(1 + 1j*self.DELTAS[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]

            self.Db_inv = self.invert_D(Db)

        E = np.einsum('nkqQ, kQ -> nq', self.Db_inv, p)
        return E
    
    def chebyshev_solverF(self, p, Y=0, G=0):
        if self.atom.splitting == True and self.protocol == 'RamanFWM':
            # need to change Db at each time step
            # notice different DELTA 
            term1 = -np.einsum('zjkbxv, zghbxvp -> zghjkvp', np.conj(Y), self.dsqrtQ)
            term2 = np.einsum('zghasv, zasbxvp -> zghjkvp', G, self.dsqrtQ)
            DbF = np.zeros(self.DbF.shape)
            DbF[:, :, :, 0] = self.DbF[:, :, :, 0] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (term1[..., 0])[..., None])/(1 + 1j*self.DELTASHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            DbF[:, :, :, 1] = self.DbF[:, :, :, 1] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (term1[..., 0])[..., None])/(1 + 1j*self.DELTASHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            DbF[:, :, :, 0] = DbF[:, :, :, 0] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (term2[..., 0])[..., None])/(1 + 1j*self.DELTASHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            DbF[:, :, :, 1] = DbF[:, :, :, 1] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQ * (term2[..., 0])[..., None])/(1 + 1j*self.DELTASHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            DbF[:, :, :, 0] = DbF[:, :, :, 0] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQF * (term1[..., 0])[..., None])/(1 + 1j*self.DELTACHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            DbF[:, :, :, 1] = DbF[:, :, :, 1] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQF * (term1[..., 0])[..., None])/(1 + 1j*self.DELTACHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]

            self.Db_inv = self.invert_D(DbF)
        elif self.atom.splitting == True and self.protocol == 'RamanFWM-Magic':
            term1 = -np.einsum('zjkbxv, zghbxvp -> zghjkvp', np.conj(Y), self.dsqrtQ)
            DbF = np.zeros(self.DbF.shape)
            DbF[:, :, :, 0] = DbF[:, :, :, 0] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQF * (term1[..., 0])[..., None])/(1 + 1j*self.DELTACHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            DbF[:, :, :, 1] = DbF[:, :, :, 1] + np.sum( np.sum( np.sum( np.sum( np.sum( ( self.dsqrtQF * (term1[..., 0])[..., None])/(1 + 1j*self.DELTACHF[None, :, None, :, None, :, None]), axis=-2), axis=-2), axis=-2), axis=-2), axis=-2)[None, :, :] * np.eye(self.n)[:, :, None]
            

        EF = np.einsum('nkqQ, kQ -> nq', self.DbF_inv, p)
        return EF

    def Butcher_tableau(self, method):
        if method == 'RK1':
            A = np.array([[0]])
            b = np.array([1])
            steps = np.array([0])
        elif method == 'RK2':
            A = np.array([[0, 0], 
                                [1/2, 0]])
            b = np.array([0, 1])
            steps = np.array([0, 1/2])
        elif method == 'RK-Heun':
            A = np.array([[0, 0], 
                                [1, 0]])
            b = np.array([0.5, 0.5])
            steps = np.array([0, 1])
        elif method == 'RK3-Heun':
            A = np.array([[0, 0, 0], 
                                [1/3, 0, 0],
                                [0, 2/3, 0]])
            b = np.array([1/4, 0, 3/4])
            steps = np.array([0, 1/3, 2/3])
        elif method == 'RK4':
            A = np.array([[0, 0, 0, 0],
                        [1/2, 0, 0, 0],
                        [0, 1/2, 0, 0],
                        [0, 0, 1, 0]])
            b = np.array([1/6, 1/3, 1/3, 1/6])
            steps = np.array([0, 1/2, 1/2, 1])
        elif method == 'RK4-3/8':
            A = np.array([[0, 0, 0, 0],
                        [1/3, 0, 0, 0],
                        [-1/3, 1, 0, 0],
                        [1, -1, 1, 0]])
            b = np.array([1/8, 3/8, 3/8, 1/8])
            steps = np.array([0, 1/3, 2/3, 1])
        elif method == 'RK4-Ralston':
            A = np.array([[0, 0, 0, 0],
                        [0.4, 0, 0, 0],
                        [0.296977651, 0.15875964, 0, 0],
                        [0.21810040, -3.05096516, 3.83286476, 0]])
            b = np.array([0.17476028, -0.55148066, 1.2055356, 0.17118478])
            steps = np.array([0, 0.4, 0.45573725, 1])
        return A, b, steps

    def calc_ei(self, ti, Control, coherences):
        opt=False
        if self.protocol == 'EIT' or self.protocol == 'FLAME':
            Pi = coherences
            p = - np.einsum('zghjkvp, zghjkv -> zp', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), Pi, optimize=opt)
            p[0] =  self.Einits(ti)
            Ei =  self.chebyshev_solver(p)
            return Ei
        elif self.protocol == 'EITFWM':
            Pi, Si = coherences
            p = - np.einsum('zghjkvp, zghjkv -> zp', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), Pi, optimize=opt)
            p[0] =  self.Einits(ti)
            Ei =  self.chebyshev_solver(p)

            p = +1j*np.einsum('zjkqwvp, zjkqwv -> zp', self.dsqrtQF,
                    np.einsum('zghjkqwv, gjv -> zjkqwv',
                        np.einsum('ghjkz, zghqwv -> zghjkqwv', 
                                  np.einsum('ghjkp,zp -> ghjkz', self.OmegaQF,  np.conj(Control(ti, self.zCheby)), optimize=opt), np.conj(Si), optimize=opt), 1/(1+1j*(self.DELTACHF)), optimize=opt), optimize=opt)
            p[0] = self.EFinits(ti)
            EFi = self.chebyshev_solverF(p)

            return Ei, EFi
        elif self.protocol == 'RamanFWM':
            Si = coherences
            p = +1j*np.einsum('zghjkvp, zghjkv -> zp', 
                          np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), 
                          np.einsum('zghjkv, gjv -> zghjkv',
                                    np.einsum('jkqwz, zghqwv-> zghjkv', 
                                              np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  Control(ti, self.zCheby), optimize=opt), Si, optimize=opt), 1/(1+1j*self.DELTAS), optimize=opt), optimize=opt)
            p[0] =  self.Einits(ti)
            Ei = self.chebyshev_solver(p)

            p = +1j*np.einsum('zjkqwvp, zjkqwv -> zp', self.dsqrtQF,
                    np.einsum('zghjkqwv, gjv -> zjkqwv',
                        np.einsum('ghjkz, zghqwv -> zghjkqwv', 
                                  np.einsum('ghjkp,zp -> ghjkz', self.OmegaQF,  np.conj(Control(ti, self.zCheby)), optimize=opt), np.conj(Si), optimize=opt), 1/(1+1j*(self.DELTACHF)), optimize=opt), optimize=opt)
            p[0] = self.EFinits(ti)
            EFi = self.chebyshev_solverF(p)

            return Ei, EFi
        elif self.protocol == 'RamanFWM-Magic':
            PFi, Si = coherences
            p = +1j*np.einsum('zghjkvp, zghjkv -> zp', 
                          np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), 
                          np.einsum('zghjkv, gjv -> zghjkv',
                                    np.einsum('jkqwz, zghqwv-> zghjkv', 
                                              np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  Control(ti, self.zCheby), optimize=opt), Si, optimize=opt), 1/(1+1j*self.DELTAS), optimize=opt), optimize=opt)
            p[0] =  self.Einits(ti)
            Ei = self.chebyshev_solver(p)

            p = (+1j*np.einsum('zjkqwvp, zjkqwv -> zp', self.dsqrtQF,
                    np.einsum('zghjkqwv, gjv -> zjkqwv',
                        np.einsum('ghjkz, zghqwv -> zghjkqwv', 
                                  np.einsum('ghjkp,zp -> ghjkz', self.OmegaQF,  np.conj(Control(ti, self.zCheby)), optimize=opt), np.conj(Si)), 1/(1+1j*(self.DELTACHF))))
                - np.einsum('zghjkvp, zghjkv -> zp', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), PFi, optimize=opt)
            )
            p[0] = self.EFinits(ti)
            EFi = self.chebyshev_solverF(p)

            return Ei, EFi
        elif self.protocol == 'TORCA' or self.protocol == 'TORCA_GSM_D1' or self.protocol == 'TORCA_GSM_D2':
            Si = coherences
            p = -1j*np.einsum('zjkqwvp, zjkqwv -> zp', 
                           self.dsqrtQ, 
                          np.einsum('zjkqwv, jqv -> zjkqwv',
                                    np.einsum('ghjkz, zghqwv-> zjkqwv', 
                                              np.einsum('ghjkz, gh -> ghjkz', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  Control(ti, self.zCheby), optimize=opt), np.sqrt(self.pop), optimize=opt), 
                                              Si, optimize=opt), 1/(self.gammaSNU + 1 + 1j*self.DELTAS), optimize=opt), optimize=opt)
            p[0] =  self.Einits(ti)
            Ei = self.chebyshev_solver(p)
            return Ei
        elif self.protocol == 'ORCA' or self.protocol == 'Raman':
            Si = coherences
            p = +1j*np.einsum('zghjkvp, zghjkv -> zp', 
                          np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), 
                          np.einsum('zghjkv, gjv -> zghjkv',
                                    np.einsum('jkqwz, zghqwv-> zghjkv', 
                                              np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  Control(ti, self.zCheby), optimize=opt), Si, optimize=opt), 1/(1+1j*self.DELTAS), optimize=opt), optimize=opt)
            
            p[0] =  self.Einits(ti)
            Ei = self.chebyshev_solver(p)
            return Ei
        elif self.protocol == 'Raman_test':
            Si, Ei = coherences
            p = (+1j*np.einsum('zghjkvp, zghjkv -> zp', 
                          np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), 
                          np.einsum('zghjkv, gjv -> zghjkv',
                                    np.einsum('jkqwz, zghqwv-> zghjkv', 
                                              np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  Control(ti, self.zCheby), optimize=opt), Si, optimize=opt), 1/(1+1j*self.DELTAS), optimize=opt), optimize=opt)

                    - np.einsum('zghjkvp, zghjkv -> zp', self.dsqrtQ, np.einsum('zghjkvp, zgjvp -> zghjkv', self.dsqrtQ, np.einsum('zp, gjv -> zgjvp', Ei, 1/(1+1j*self.DELTAS))))
                )            
            p[0] =  self.Einits(ti)
            Ei = self.chebyshev_solver(p)
            return Ei
        elif self.protocol == 'TORCAP' or self.protocol == '4levelTORCAP' or self.protocol == 'TORCAP_2dressing_states':
            Pesi = coherences 
            p = - np.einsum('zjkqwvp, zjkqwv -> zp', self.dsqrtQ, Pesi)
            p[0] =  self.Einits(ti)
            Ei = self.chebyshev_solver(p)
            return Ei
        elif self.protocol == 'ORCA_GSM':
            Pge1, Pge2 = coherences
            p = - np.einsum('zghjkvp, zghjkv -> zp', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), Pge1)
            p[0] =  self.Einits(ti)
            Eis = self.chebyshev_solver(p)
            p = - np.einsum('zghjkvp, zghjkv -> zp', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), Pge2)
            p[0] =  0 #self.Einits(ti)
            EiR = self.chebyshev_solver(p)
            return Eis, EiR


    def RK(self, Control):
        if self.protocol == 'EIT' or self.protocol == 'FLAME':
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KP = np.zeros((len(self.steps), *self.P[0].shape) , dtype=complex) #to hold intermediate values
            self.KS = np.zeros((len(self.steps), *self.S[0].shape) , dtype=complex) #to hold intermediate values

            self.E[0] = self.calc_ei(0, Control, (self.P[0]))
            for mi in range(1, self.m):
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Pi = self.P[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KP, axis=0)
                    Si = self.S[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KS, axis=0)
                    self.KE[ki] = self.calc_ei(ti, Control, (Pi))
                    self.KP[ki], self.KS[ki] = ( self.Pderivative(ti, ( Pi,  Si, self.KE[ki]), Control),
                                                self.Sderivative(ti, ( Pi,  Si), Control) )
                
                self.P[mi], self.S[mi] = ( self.P[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KP, axis=0),
                                            self.S[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KS, axis=0))
                
                self.E[mi] = self.calc_ei(ti, Control, (self.P[mi]))
                
                self.KE[:] = 0  #empty K
                self.KP[:] = 0
                self.KS[:] = 0

            
        elif self.protocol == 'EITFWM':
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KEF = np.zeros((len(self.steps), *self.EF[0].shape) , dtype=complex) #to hold intermediate values
            self.KP = np.zeros((len(self.steps), *self.P[0].shape) , dtype=complex) #to hold intermediate values
            self.KS = np.zeros((len(self.steps), *self.S[0].shape) , dtype=complex) #to hold intermediate values

            self.E[0], self.EF[0] = self.calc_ei(0, Control, (self.P[0], self.S[0]))
            for mi in range(1, self.m):
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Pi = self.P[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KP, axis=0)
                    Si = self.S[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KS, axis=0)
                    self.KE[ki], self.KEF[ki] = self.calc_ei(ti, Control, (Pi, Si))
                    self.KP[ki], self.KS[ki] = ( self.Pderivative(ti, ( Pi,  Si, self.KE[ki]), Control),
                                                self.Sderivative(ti, ( Pi,  Si, self.KEF[ki]), Control) )
                
                self.P[mi], self.S[mi] = ( self.P[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KP, axis=0),
                                            self.S[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KS, axis=0))
                
                self.E[mi], self.EF[mi] = self.calc_ei(ti, Control, (self.P[mi], self.S[mi]))
                
                self.KE[:] = 0  #empty K
                self.KEF[:] = 0
                self.KP[:] = 0
                self.KS[:] = 0

            
        elif self.protocol == 'RamanFWM':
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KEF = np.zeros((len(self.steps), *self.EF[0].shape) , dtype=complex) #to hold intermediate values
            self.KYee = np.zeros((len(self.steps), *self.Yee[0].shape) , dtype=complex)
            self.KMgg = np.zeros((len(self.steps), *self.Mgg[0].shape) , dtype=complex) 
            self.KS = np.zeros((len(self.steps), *self.S[0].shape) , dtype=complex) #to hold intermediate values

            self.E[0], self.EF[0] = self.calc_ei(0, Control, (self.S[0]))
            for mi in range(1, self.m):
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Yeei = self.Yee[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KYee, axis=0)
                    Mggi = self.Mgg[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KMgg, axis=0)
                    Si = self.S[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KS, axis=0)
                    self.KE[ki], self.KEF[ki] = self.calc_ei(ti, Control, (Si))
                    self.KYee[ki] = ( self.Yderivative(ti, (Si, self.KE[ki], Yeei, Mggi), Control) )
                    self.KMgg[ki] = ( self.Mderivative(ti, (Si, self.KE[ki], Yeei, Mggi), Control) )
                    self.KS[ki] = ( self.Sderivative(ti, ( Si, self.KE[ki], self.KEF[ki], Yeei, Mggi), Control) )
                
                self.Yee[mi] = ( self.Yee[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KYee, axis=0))
                self.Mgg[mi] = ( self.Mgg[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KMgg, axis=0))
                self.S[mi] = ( self.S[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KS, axis=0))
                
                self.E[mi], self.EF[mi] = self.calc_ei(ti, Control, (self.S[mi]))
                
                self.KE[:] = 0  #empty K
                self.KEF[:] = 0
                self.KYee[:] = 0
                self.KMgg[:] = 0
                self.KS[:] = 0

            
        elif self.protocol == 'RamanFWM-Magic':
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KEF = np.zeros((len(self.steps), *self.EF[0].shape) , dtype=complex) #to hold intermediate values
            self.KPF = np.zeros((len(self.steps), *self.PF[0].shape) , dtype=complex) #to hold intermediate values
            self.KYee = np.zeros((len(self.steps), *self.Yee[0].shape) , dtype=complex)
            self.KMgg = np.zeros((len(self.steps), *self.Mgg[0].shape) , dtype=complex) 
            self.KS = np.zeros((len(self.steps), *self.S[0].shape) , dtype=complex) #to hold intermediate values

            self.E[0], self.EF[0] = self.calc_ei(0, Control, (self.PF[0], self.S[0]))
            for mi in range(1, self.m):
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    PFi = self.PF[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KPF, axis=0)
                    Yeei = self.Yee[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KYee, axis=0)
                    Mggi = self.Mgg[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KMgg, axis=0)
                    Si = self.S[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KS, axis=0)
                    self.KE[ki], self.KEF[ki] = self.calc_ei(ti, Control, (PFi, Si))
                    self.KYee[ki] = ( self.Yderivative(ti, (Si, self.KE[ki], Yeei, Mggi), Control) )
                    self.KMgg[ki] = ( self.Mderivative(ti, (Si, self.KE[ki], Yeei, Mggi), Control) )
                    self.KPF[ki], self.KS[ki] = ( self.Pderivative(ti, ( PFi,  self.KEF[ki], Yeei, Mggi), Control),
                                                self.Sderivative(ti, ( Si, self.KE[ki], self.KEF[ki], Yeei, Mggi), Control) )
                
                self.PF[mi], self.S[mi] = ( self.PF[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KPF, axis=0),
                                            self.S[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KS, axis=0))
                self.Yee[mi] = ( self.Yee[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KYee, axis=0))
                self.Mgg[mi] = ( self.Mgg[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KMgg, axis=0))
                
                self.E[mi], self.EF[mi] = self.calc_ei(ti, Control, (self.PF[mi], self.S[mi]))
                
                self.KE[:] = 0  #empty K
                self.KEF[:] = 0
                self.KPF[:] = 0
                self.KYee[:] = 0
                self.KMgg[:] = 0
                self.KS[:] = 0

            
        elif self.protocol == 'TORCA' or self.protocol == 'ORCA' or self.protocol == 'Raman':
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KS = np.zeros((len(self.steps), *self.S[0].shape) , dtype=complex) #to hold intermediate values

            self.E[0] = self.calc_ei(0, Control, (self.S[0]))
            for mi in range(1, self.m):
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Si = self.S[mi-1] + self.tstep*np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KS, axis=0)
                    self.KE[ki] = self.calc_ei(ti, Control, (Si))
                    self.KS[ki] = ( self.Sderivative(ti, ( Si, self.KE[ki]), Control) )
                
                self.S[mi] = ( self.S[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KS, axis=0))
                
                self.E[mi] = self.calc_ei(ti, Control, (self.S[mi]))
                
                self.KE[:] = 0  #empty K
                self.KS[:] = 0

        elif self.protocol == 'Raman_test':
            self.KE = np.zeros((len(self.steps)+1, *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KS = np.zeros((len(self.steps), *self.S[0].shape) , dtype=complex) #to hold intermediate values
            self.E[0] = self.calc_ei(0, Control, (self.S[0], np.zeros((self.n, 2))))
            for mi in range(1, self.m):
                self.KE[0] = self.E[mi-1]
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Si = self.S[mi-1] + self.tstep*np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KS, axis=0)
                    self.KE[ki+1] = self.calc_ei(ti, Control, (Si, self.KE[ki]))
                    self.KS[ki] = ( self.Sderivative(ti, ( Si, self.KE[ki+1]), Control) )
                
                self.S[mi] = ( self.S[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KS, axis=0))
                
                self.E[mi] = self.calc_ei(ti, Control, (self.S[mi], self.KE[-1]))
                
                self.KE[:] = 0  #empty K
                self.KS[:] = 0

            
        elif self.protocol == 'TORCAP':
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KPge = np.zeros((len(self.steps), *self.Pge[0].shape) , dtype=complex) #to hold intermediate values
            self.KPes = np.zeros((len(self.steps), *self.Pes[0].shape) , dtype=complex) #to hold intermediate values
            self.KYee = np.zeros((len(self.steps), *self.Yee[0].shape) , dtype=complex)
            self.KMgg = np.zeros((len(self.steps), *self.Mgg[0].shape) , dtype=complex) 
            self.KS = np.zeros((len(self.steps), *self.S[0].shape) , dtype=complex) #to hold intermediate values

            self.E[0] = self.calc_ei(0, Control, (self.Pes[0]))
            for mi in range(1, self.m):
                print(mi)
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Yeei = self.Yee[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KYee, axis=0)
                    Mggi = self.Mgg[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KMgg, axis=0)
                    Si = self.S[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KS, axis=0)
                    self.KPge[ki], self.KPes[ki] = self.adiabaticP(ti, (Yeei, Mggi, Si, self.KE[ki]), Control)
                    self.KE[ki] = self.calc_ei(ti, Control, (self.KPes[ki]))
                    self.KYee[ki] = ( self.Yderivative(ti, (self.KPge[ki], self.KPes[ki], Yeei, self.KE[ki]), Control) )
                    self.KMgg[ki] = ( self.Mderivative(ti, (self.KPge[ki]), Control) )
                    self.KS[ki] = ( self.Sderivative(ti, (self.KPge[ki], self.KPes[ki], Si, self.KE[ki]), Control) )
                
                self.Yee[mi] = ( self.Yee[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KYee, axis=0))
                self.Mgg[mi] = ( self.Mgg[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KMgg, axis=0))
                self.S[mi] = ( self.S[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KS, axis=0))
                self.Pge[mi], self.Pes[mi] = self.adiabaticP(ti, (self.Yee[mi], self.Mgg[mi], self.S[mi], self.E[mi-1]), Control)
                self.E[mi] = self.calc_ei(ti, Control, (self.Pes[mi]))
                
                self.KE[:] = 0  #empty K
                self.KPge[:] = 0
                self.KPes[:] = 0
                self.KYee[:] = 0
                self.KMgg[:] = 0
                self.KS[:] = 0
            
            
        elif self.protocol == '4levelTORCAP':
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KPge = np.zeros((len(self.steps), *self.Pge[0].shape) , dtype=complex) #to hold intermediate values
            self.KPes = np.zeros((len(self.steps)+1, *self.Pes[0].shape) , dtype=complex) #to hold intermediate values
            self.KPeb = np.zeros((len(self.steps)+1, *self.Peb[0].shape) , dtype=complex) #to hold intermediate values
            self.KYee = np.zeros((len(self.steps), *self.Yee[0].shape) , dtype=complex)
            self.KMgg = np.zeros((len(self.steps), *self.Mgg[0].shape) , dtype=complex) 
            self.KSgs = np.zeros((len(self.steps), *self.Sgs[0].shape) , dtype=complex) #to hold intermediate values
            self.KSgb = np.zeros((len(self.steps), *self.Sgb[0].shape) , dtype=complex) #to hold intermediate values
            
            self.E[0] = self.calc_ei(0, Control, (self.Pes[0]))
            for mi in range(1, self.m):
                self.KPes[0] = self.Pes[mi-1]
                self.KPeb[0] = self.Peb[mi-1]
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Yeei = self.Yee[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KYee, axis=0)
                    Mggi = self.Mgg[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KMgg, axis=0)
                    Sgsi = self.Sgs[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgs, axis=0)
                    Sgbi = self.Sgb[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgb, axis=0)
                    self.KPge[ki], self.KPes[ki+1], self.KPeb[ki+1] = self.adiabaticP(ti, (Yeei, Mggi, self.KPes[ki], self.KPeb[ki], Sgsi, Sgbi, self.KE[ki]), Control)
                    self.KE[ki] = self.calc_ei(ti, Control, (self.KPes[ki+1]))
                    self.KYee[ki] = ( self.Yderivative(ti, (self.KPge[ki], self.KPes[ki], Yeei, self.KE[ki]), Control) )
                    self.KMgg[ki] = ( self.Mderivative(ti, (self.KPge[ki]), Control) )
                    self.KSgs[ki], self.KSgb[ki] = ( self.Sderivative(ti, (self.KPge[ki],  self.KPes[ki+1], self.KPeb[ki+1], Sgsi, Sgbi, self.KE[ki]), Control) )
                
                self.Yee[mi] = ( self.Yee[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KYee, axis=0))
                self.Mgg[mi] = ( self.Mgg[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KMgg, axis=0))
                self.Sgs[mi] = ( self.Sgs[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgs, axis=0))
                self.Sgb[mi] = ( self.Sgb[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgb, axis=0))
                self.Pge[mi], self.Pes[mi], self.Peb[mi] = self.adiabaticP(ti, (self.Yee[mi], self.Mgg[mi], self.Pes[mi-1], self.Peb[mi-1], self.Sgs[mi], self.Sgb[mi], self.E[mi-1]), Control)
                self.E[mi] = self.calc_ei(ti, Control, (self.Pes[mi]))
                
                self.KE[:] = 0  #empty K
                self.KPge[:] = 0
                self.KPes[:] = 0
                self.KPeb[:] = 0
                self.KYee[:] = 0
                self.KMgg[:] = 0
                self.KSgs[:] = 0
                self.KSgb[:] = 0
            
            
        elif self.protocol == 'TORCAP_2dressing_states':
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KPge = np.zeros((len(self.steps), *self.Pge[0].shape) , dtype=complex) #to hold intermediate values
            self.KPes = np.zeros((len(self.steps)+1, *self.Pes[0].shape) , dtype=complex) #to hold intermediate values
            self.KPeb = np.zeros((len(self.steps)+1, *self.Peb[0].shape) , dtype=complex) #to hold intermediate values
            self.KPeb2 = np.zeros((len(self.steps)+1, *self.Peb2[0].shape) , dtype=complex) #to hold intermediate values
            self.KYee = np.zeros((len(self.steps), *self.Yee[0].shape) , dtype=complex)
            self.KMgg = np.zeros((len(self.steps), *self.Mgg[0].shape) , dtype=complex) 
            self.KSgs = np.zeros((len(self.steps), *self.Sgs[0].shape) , dtype=complex) #to hold intermediate values
            self.KSgb = np.zeros((len(self.steps), *self.Sgb[0].shape) , dtype=complex) #to hold intermediate values
            self.KSgb2 = np.zeros((len(self.steps), *self.Sgb2[0].shape) , dtype=complex) #to hold intermediate values
            
            self.E[0] = self.calc_ei(0, Control, (self.Pes[0]))
            for mi in range(1, self.m):
                self.KPes[0] = self.Pes[mi-1]
                self.KPeb[0] = self.Peb[mi-1]
                self.KPeb2[0] = self.Peb2[mi-1]
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Yeei = self.Yee[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KYee, axis=0)
                    Mggi = self.Mgg[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KMgg, axis=0)
                    Sgsi = self.Sgs[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgs, axis=0)
                    Sgbi = self.Sgb[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgb, axis=0)
                    Sgb2i = self.Sgb2[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgb2, axis=0)
                    self.KPge[ki], self.KPes[ki+1], self.KPeb[ki+1], self.KPeb2[ki+1] = self.adiabaticP(ti, (Yeei, Mggi, self.KPes[ki], self.KPeb[ki], self.KPeb2[ki], Sgsi, Sgbi, Sgb2i, self.KE[ki]), Control)
                    self.KE[ki] = self.calc_ei(ti, Control, (self.KPes[ki+1]))
                    self.KYee[ki] = ( self.Yderivative(ti, (self.KPge[ki], self.KPes[ki], Yeei, self.KE[ki]), Control) )
                    self.KMgg[ki] = ( self.Mderivative(ti, (self.KPge[ki]), Control) )
                    self.KSgs[ki], self.KSgb[ki], self.KSgb2[ki] = ( self.Sderivative(ti, (self.KPge[ki],  self.KPes[ki+1], self.KPeb[ki+1], self.KPeb2[ki+1], Sgsi, Sgbi, Sgb2i, self.KE[ki]), Control) )
                
                self.Yee[mi] = ( self.Yee[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KYee, axis=0))
                self.Mgg[mi] = ( self.Mgg[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KMgg, axis=0))
                self.Sgs[mi] = ( self.Sgs[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgs, axis=0))
                self.Sgb[mi] = ( self.Sgb[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgb, axis=0))
                self.Sgb2[mi] = ( self.Sgb2[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgb2, axis=0))
                self.Pge[mi], self.Pes[mi], self.Peb[mi], self.Peb2[mi] = self.adiabaticP(ti, (self.Yee[mi], self.Mgg[mi], self.Pes[mi-1], self.Peb[mi-1], self.Peb2[mi-1], self.Sgs[mi], self.Sgb[mi], self.Sgb2[mi], self.E[mi-1]), Control)
                self.E[mi] = self.calc_ei(ti, Control, (self.Pes[mi]))
                
                self.KE[:] = 0  #empty K
                self.KPge[:] = 0
                self.KPes[:] = 0
                self.KPeb[:] = 0
                self.KPeb2[:] = 0
                self.KYee[:] = 0
                self.KMgg[:] = 0
                self.KSgs[:] = 0
                self.KSgb[:] = 0
                self.KSgb2[:] = 0
            
            
        elif self.protocol == 'ORCA_GSM':
            # self.KE[mi-1] to work out adiabatic, then workout E? Is this the correct order
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KER = np.zeros((len(self.steps), *self.ER[0].shape) , dtype=complex) #to hold intermediate values
            self.KPge = np.zeros((len(self.steps)+1, *self.Pge[0].shape) , dtype=complex) #to hold intermediate values
            self.KPge2 = np.zeros((len(self.steps), *self.Pge2[0].shape) , dtype=complex) #to hold intermediate values
            self.KSgs = np.zeros((len(self.steps), *self.Sgs[0].shape) , dtype=complex) #to hold intermediate values
            self.KSgb = np.zeros((len(self.steps), *self.Sgb[0].shape) , dtype=complex) #to hold intermediate values

            self.E[0], self.ER[0] = self.calc_ei(0, Control, (self.Pge[0], self.Pge2[0]))
            for mi in range(1, self.m):
                self.KPge[0] = self.Pge[mi-1]
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Sgsi = self.Sgs[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgs, axis=0)
                    Sgbi = self.Sgb[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgb, axis=0)
                    Pge2i = self.Pge2[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KPge2, axis=0)

                    self.KE[ki], self.KER[ki] = self.calc_ei(ti, Control, (self.KPge[ki], Pge2i))
                    self.KPge[ki+1] = self.adiabaticP(ti, (Sgsi, self.KE[ki]), Control)
                    self.KPge2[ki] = self.Pderivative(ti, (Pge2i, Sgsi, Sgbi, self.KER[ki]), Control)
                    self.KSgs[ki], self.KSgb[ki] = ( self.Sderivative(ti, (self.KPge[ki+1], Pge2i, Sgsi, Sgbi), Control) )
                    
                    #self.KE[ki], self.KER[ki] = self.calc_ei(ti, Control, (self.Pge[mi-1], Pge2i))

                    #self.KPge2[ki] = self.Pderivative(ti, (Pge2i, Sgsi, Sgbi, self.KER[ki]), Control)
                    #self.KSgs[ki], self.KSgb[ki] = ( self.Sderivative(ti, (self.Pge[mi-1], Pge2i, Sgsi, Sgbi), Control) )
                    

                self.Sgs[mi] = ( self.Sgs[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgs, axis=0))
                self.Sgb[mi] = ( self.Sgb[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgb, axis=0))
                self.Pge2[mi] = ( self.Pge2[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KPge2, axis=0))
                self.Pge[mi] = self.adiabaticP(ti, (self.Sgs[mi], self.E[mi-1]), Control)
                self.E[mi], self.ER[mi] = self.calc_ei(ti, Control, (self.Pge[mi], self.Pge2[mi]))

                #self.Pge[mi] = self.adiabaticP(ti, (self.Sgs[mi-1], self.E[mi-1]), Control)
                #self.Sgs[mi] = ( self.Sgs[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgs, axis=0))
                #self.Sgb[mi] = ( self.Sgb[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgb, axis=0))
                #self.Pge2[mi] = ( self.Pge2[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KPge2, axis=0))
                
                #self.E[mi], self.ER[mi] = self.calc_ei(ti, Control, (self.Pge[mi], self.Pge2[mi]))
                
                self.KE[:] = 0  #empty K
                self.KER[:] = 0
                self.KPge[:] = 0
                self.KPge2[:] = 0
                self.KSgs[:] = 0
                self.KSgb[:] = 0

            

        #elif self.protocol == '4levelORCA':
            # self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            # self.KSgs = np.zeros((len(self.steps), *self.Sgs[0].shape) , dtype=complex) #to hold intermediate values
            # self.KSgb = np.zeros((len(self.steps), *self.Sgb[0].shape) , dtype=complex) #to hold intermediate values   

        elif self.protocol == 'TORCA_GSM_D1' or self.protocol == 'TORCA_GSM_D2':
            self.Pge2 = np.zeros((self.m, self.n, len(self.Fg), len(self.mg), len(self.Fb), len(self.mb), self.vno), dtype=complex)
            self.Sgs = np.zeros((self.m, self.n, len(self.Fg), len(self.mg), len(self.Fq), len(self.mq), self.vno), dtype=complex) #(t, z, number of ground states, number of mF states in ground state with largest F, number of storage states, number of mF states in storage state with largest F, number of velocity classes)
            self.Sgs2 = np.zeros((self.m, self.n, len(self.Fg), len(self.mg), len(self.Fh), len(self.mh), self.vno), dtype=complex)
            
            self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            self.KPge2 = np.zeros((len(self.steps), *self.P[0].shape) , dtype=complex) #to hold intermediate values
            self.KSgs = np.zeros((len(self.steps), *self.Sgs[0].shape) , dtype=complex) #to hold intermediate values
            self.KSgs2 = np.zeros((len(self.steps), *self.Sgb[0].shape) , dtype=complex) #to hold intermediate values
            
            self.E[0] = self.calc_ei(0, Control, (self.Sgs[0]))
            for mi in range(1, self.m):
                for ki in range(len(self.steps)):
                    dt = self.tstep*self.steps[ki]
                    ti = self.tpoints[mi-1] + dt
                    Pge2i = self.Pge2[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KPge2, axis=0)
                    Sgsi = self.Sgs[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgs, axis=0)
                    Sgs2i = self.Sgs2[mi-1] + self.tstep * np.sum(self.A[ki, :, None, None, None, None, None, None] * self.KSgs2, axis=0)
                    self.KE[ki] = self.calc_ei(ti, Control, (Sgsi))
                    self.KPge2[ki] = self.Pderivative(ti, (Pge2i, Sgsi, Sgs2i), Control)
                    self.KSgs[ki], self.KSgs2[ki] = ( self.Sderivative(ti, (Pge2i, Sgsi, Sgs2i, self.KE[ki]), Control) )
                
                self.Sgs[mi] = ( self.Sgs[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgs, axis=0))
                self.Sgs2[mi] = ( self.Sgs2[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KSgs2, axis=0))
                self.Pge2[mi] = ( self.Pge2[mi-1] + self.tstep * np.sum(self.b[:, None, None, None, None, None, None] * self.KPge2, axis=0))
                self.E[mi] = self.calc_ei(ti, Control, (self.Sgs[mi]))
                
                self.KE[:] = 0  #empty K
                self.KPge2[:] = 0
                self.KSgs[:] = 0
                self.KSgs2[:] = 0
            
            
        #elif self.protocol == '4levelORCA':
            # self.KE = np.zeros((len(self.steps), *self.E[0].shape) , dtype=complex) #to hold intermediate values
            # self.KSgs = np.zeros((len(self.steps), *self.Sgs[0].shape) , dtype=complex) #to hold intermediate values
            # self.KSgb = np.zeros((len(self.steps), *self.Sgb[0].shape) , dtype=complex) #to hold intermediate values   

    def solve(self, Control, method='RK4'):
        self.A, self.b, self.steps = self.Butcher_tableau(method)
        try:

            self.RK(Control)
            self.solved = True

            if self.protocol != 'RamanFWM' or self.protocol != 'RamanFWM-Magic' or self.protocol != 'EITFWM':
                total = self.check_energy()
            if total > 1 * self.energy_tol: #?
                self.solved = False
            
        except:
            self.solved = False

    def Pderivative(self, ti, coherences, Control):
        # (g, mg, j, mj, q, mq, v, z, Q)
        opt=False
        if self.protocol == 'EIT' or self.protocol == 'FLAME' or self.protocol == 'EITFWM':
            P, S, E = coherences            
            dP = ( np.einsum('gjv, zghjkv -> zghjkv', -(1+1j*self.DELTAS), P, optimize=opt)
                  + np.einsum('zghjkvp, zp -> zghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E, optimize=opt)          
                  -1j*np.einsum('zjkqw, zghqwv -> zghjkv', np.einsum('jkqwp, zp -> zjkqw', self.OmegaQ, Control(ti, self.zCheby), optimize=opt), S, optimize=opt) 
                )
        elif self.protocol == 'RamanFWM-Magic':
            PF, EF, Y, M = coherences
            dP = ( np.einsum('gjv, zghjkv -> zghjkv', -(1+1j*self.DELTASHF), PF, optimize=opt)
                   + np.einsum('zghjkvp, zp -> zghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), EF, optimize=opt)
                   - np.einsum('zjkbxv, zghbxv -> zghjkv', np.conj(Y), np.einsum('zghbxvp, zp -> zghbxv', np.einsum('zghbxvp, gh -> zghbxvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), EF, optimize=opt))
                   + np.einsum('zghasv, zasjkv -> zghjkv', M, np.einsum('zasjkvp, zp -> zasjkv', np.einsum('zasjkvp, as -> zasjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), EF, optimize=opt))
            )
        elif self.protocol == 'ORCA_GSM':
            P, Sgs, Sgb, ER = coherences
            #Control0 = Control[0]
            M1 = Control[1]
            M2 = Control[2]
            #Control_zp0 = Control0(ti, self.zCheby)
            M1_zp = M1(ti, self.zCheby)
            M2_zp = M2(ti, self.zCheby)
            dP =( np.einsum('gbv, zghbxv -> zghbxv', -(1+1j*self.DELTAGJ), P, optimize=opt)  
                 + np.einsum('zghjkvp, zp -> zghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), ER, optimize=opt)          
                        -1j*np.einsum('zjkqw, zghqwv -> zghjkv', np.einsum('jkqwp, zp -> zjkqw', self.OmegaQ, M1_zp, optimize=opt), Sgs, optimize=opt)
                        -1j*np.einsum('zjkbx, zghbxv -> zghjkv', np.einsum('bxjkp, zp -> zjkbx', self.OmegaM2Q, M2_zp, optimize=opt), Sgb, optimize=opt)
                        )
            return dP
        elif self.protocol == 'TORCA_GSM_D1' or self.protocol == 'TORCA_GSM_D2':
            P, S, S2 = coherences            
            [Control1, Control2, Control3] = Control 
            #Control_zp1 = Control1(ti, self.zCheby)
            Control_zp2 = Control2(ti, self.zCheby)
            Control_zp3 = Control3(ti, self.zCheby)
            dP = ( np.einsum('gbv, zghbxv -> zghbxv', -(self.gammaBNU+1j*self.DELTAGB), P, optimize=opt)        
                  -1j*np.einsum('zqwbx, zghqwv -> zghbxv', np.einsum('jqwbxp, zp -> zqwbx', self.OmegaQ2, Control_zp2, optimize=opt), S, optimize=opt) 
                  -1j*np.einsum('zbxay, zghayv -> zghbxv', np.einsum('jbxayp, zp -> zbxay', self.OmegaQ3, Control_zp3, optimize=opt), S2, optimize=opt) 
                )
        return dP
    
    def adiabaticP(self, ti, coherences, Control):
        opt=False
        if self.protocol == 'TORCAP':
            Y, M, S, E = coherences
            Control_zp = Control(ti, self.zCheby)
            Pge = np.einsum('zghjk, gjv -> zghjkv', 
                                
                                -1j*np.einsum('zghjk, gh -> zghjk',
                                    np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, Control_zp, optimize=opt), 
                                    np.sqrt(self.pop), optimize=opt)

                                +1j*np.einsum('zghbx, zjkbxv -> zghjk',
                                    np.einsum('ghbxp, zp -> zghbx', self.OmegaQ, Control_zp, optimize=opt), 
                                    np.conj(Y), optimize=opt)

                                -1j*np.einsum('zasjk, zghasv -> zghjk',
                                    np.einsum('asjkp, zp -> zasjk', self.OmegaQ, Control_zp, optimize=opt), 
                                    M, optimize=opt)

                                , 1/(1 + 1j*self.DELTAC), optimize=opt)
            
            Pes = np.einsum('zjkqwv, jqv-> zjkqwv', 
                                
                                +1j*np.einsum('zghjk, zghqwv -> zjkqwv', 
                                    np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, np.conj(Control_zp) , optimize=opt), 
                                    S, optimize=opt)  

                                +   np.einsum('zjkbxv, zbxqwv -> zjkqwv', Y,
                                    np.einsum('zbxqwvp, zp -> zbxqwv', self.dsqrtQ, E, optimize=opt) )
                                
                                , 1/(1 + self.gammaSNU + 1j*self.DELTAS), optimize=opt)
            return Pge, Pes
        elif self.protocol == '4levelTORCAP':
            Y, M, Pesi, Pebi, Sgs, Sgb, E = coherences
            Control1 = Control[0]
            Control2 = Control[1]
            Control_zp1 = Control1(ti, self.zCheby)
            Control_zp2 = Control2(ti, self.zCheby)

            Pge = np.einsum('zghjk, gjv -> zghjkv', 
                                
                                -1j*np.einsum('zghjk, gh -> zghjk',
                                    np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, Control_zp1, optimize=opt), 
                                    np.sqrt(self.pop), optimize=opt)

                                +1j*np.einsum('zghbx, zjkbxv -> zghjk',
                                    np.einsum('ghbxp, zp -> zghbx', self.OmegaQ, Control_zp1, optimize=opt), 
                                    np.conj(Y), optimize=opt)

                                -1j*np.einsum('zasjk, zghasv -> zghjk',
                                    np.einsum('asjkp, zp -> zasjk', self.OmegaQ, Control_zp1, optimize=opt), 
                                    M, optimize=opt)

                                , 1/(1 + 1j*self.DELTAC), optimize=opt)

            Pes = np.einsum('zjkqwv, jqv-> zjkqwv', 
                                
                                +1j*np.einsum('zghjk, zghqwv -> zjkqwv', 
                                    np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, np.conj(Control_zp1) , optimize=opt), 
                                    Sgs, optimize=opt)  

                                +   np.einsum('zjkbxv, zbxqwv -> zjkqwv', Y,
                                    np.einsum('zbxqwvp, zp -> zbxqwv', self.dsqrtQ, E, optimize=opt) )

                                -1j*np.einsum('zqwbx, zjkbxv -> zjkqwv', 
                                    np.einsum('qwbxp, zp -> zqwbx', self.OmegaQ2, np.conj(Control_zp2) , optimize=opt), 
                                    Pebi, optimize=opt)
                                
                                , 1/(1 + self.gammaSNU + 1j*self.DELTAS), optimize=opt)

            # Pes = (+1j*np.einsum('zjkqwv, jqv-> zjkqwv', 
            #                     np.einsum('zghjk, zghqwv -> zjkqwv', 
            #                     np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, np.conj(Control_zp1) , optimize=opt), 
            #                     Sgs, optimize=opt),  1/(1 + self.gammaSNU + 1j*self.DELTAS), optimize=opt)
                            
            #                 -1j*np.einsum('zjkqwv, jqv-> zjkqwv', 
            #                         np.einsum('zqwbx, zjkbxv -> zjkqwv', 
            #                         np.einsum('qwbxp, zp -> zqwbx', self.OmegaQ2, np.conj(Control_zp2) , optimize=opt), 
            #                         Pebi, optimize=opt),  1/(1 + self.gammaSNU + 1j*self.DELTAS), optimize=opt)
            #             )

            Peb = (+1j*np.einsum('zjkbxv, jbv-> zjkbxv', 
                                np.einsum('zghjk, zghbxv -> zjkbxv', 
                                np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, np.conj(Control_zp1) , optimize=opt), 
                                Sgb, optimize=opt),  1/(1 + self.gammaBNU + 1j*self.DELTAEB), optimize=opt)
                            
                            -1j*np.einsum('zjkbxv, jbv-> zjkbxv', 
                                    np.einsum('zqwbx, zjkqwv -> zjkbxv', 
                                    np.einsum('qwbxp, zp -> zqwbx', self.OmegaQ2, Control_zp2, optimize=opt), 
                                    Pesi, optimize=opt),  1/(1 + self.gammaBNU + 1j*self.DELTAEB), optimize=opt)
                        )
            return Pge, Pes, Peb
        elif self.protocol == 'TORCAP_2dressing_states':
            Y, M, Pesi, Pebi, Pebi2, Sgs, Sgb, Sgb2, E = coherences
            Control1 = Control[0]
            Control2 = Control[1]
            Control_zp1 = Control1(ti, self.zCheby)
            Control_zp2 = Control2(ti, self.zCheby)
            Control_zp3 = self.rabi_modification*Control_zp2

            Pge = np.einsum('zghjk, gjv -> zghjkv', 
                                
                                -1j*np.einsum('zghjk, gh -> zghjk',
                                    np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, Control_zp1, optimize=opt), 
                                    np.sqrt(self.pop), optimize=opt)

                                +1j*np.einsum('zghbx, zjkbxv -> zghjk',
                                    np.einsum('ghbxp, zp -> zghbx', self.OmegaQ, Control_zp1, optimize=opt), 
                                    np.conj(Y), optimize=opt)

                                -1j*np.einsum('zasjk, zghasv -> zghjk',
                                    np.einsum('asjkp, zp -> zasjk', self.OmegaQ, Control_zp1, optimize=opt), 
                                    M, optimize=opt)

                                , 1/(1 + 1j*self.DELTAC), optimize=opt)

            Pes = np.einsum('zjkqwv, jqv-> zjkqwv', 
                                
                                +1j*np.einsum('zghjk, zghqwv -> zjkqwv', 
                                    np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, np.conj(Control_zp1) , optimize=opt), 
                                    Sgs, optimize=opt)  

                                +   np.einsum('zjkbxv, zbxqwv -> zjkqwv', Y,
                                    np.einsum('zbxqwvp, zp -> zbxqwv', self.dsqrtQ, E, optimize=opt) )

                                -1j*np.einsum('zqwbx, zjkbxv -> zjkqwv', 
                                    np.einsum('qwbxp, zp -> zqwbx', self.OmegaQ2, np.conj(Control_zp2) , optimize=opt), 
                                    Pebi, optimize=opt)

                                -1j*np.einsum('zqwbx, zjkbxv -> zjkqwv', 
                                    np.einsum('qwbxp, zp -> zqwbx', self.OmegaQ3, np.conj(Control_zp3) , optimize=opt), 
                                    Pebi2, optimize=opt)
                                
                                , 1/(1 + self.gammaSNU + 1j*self.DELTAS), optimize=opt)

            # Pes = (+1j*np.einsum('zjkqwv, jqv-> zjkqwv', 
            #                     np.einsum('zghjk, zghqwv -> zjkqwv', 
            #                     np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, np.conj(Control_zp1) , optimize=opt), 
            #                     Sgs, optimize=opt),  1/(1 + self.gammaSNU + 1j*self.DELTAS), optimize=opt)
                            
            #                 -1j*np.einsum('zjkqwv, jqv-> zjkqwv', 
            #                         np.einsum('zqwbx, zjkbxv -> zjkqwv', 
            #                         np.einsum('qwbxp, zp -> zqwbx', self.OmegaQ2, np.conj(Control_zp2) , optimize=opt), 
            #                         Pebi, optimize=opt),  1/(1 + self.gammaSNU + 1j*self.DELTAS), optimize=opt)
            #             )

            Peb = (+1j*np.einsum('zjkbxv, jbv-> zjkbxv', 
                                np.einsum('zghjk, zghbxv -> zjkbxv', 
                                np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, np.conj(Control_zp1) , optimize=opt), 
                                Sgb, optimize=opt),  1/(1 + self.gammaBNU + 1j*self.DELTAEB), optimize=opt)
                            
                            -1j*np.einsum('zjkbxv, jbv-> zjkbxv', 
                                    np.einsum('zqwbx, zjkqwv -> zjkbxv', 
                                    np.einsum('qwbxp, zp -> zqwbx', self.OmegaQ2, Control_zp2, optimize=opt), 
                                    Pesi, optimize=opt),  1/(1 + self.gammaBNU + 1j*self.DELTAEB), optimize=opt)
                        )
            
            Peb2 = (+1j*np.einsum('zjkbxv, jbv-> zjkbxv', 
                                np.einsum('zghjk, zghbxv -> zjkbxv', 
                                np.einsum('ghjkp, zp -> zghjk', self.OmegaQ, np.conj(Control_zp1) , optimize=opt), 
                                Sgb2, optimize=opt),  1/(1 + self.gammaBNU + 1j*self.DELTAEB2), optimize=opt)
                            
                            -1j*np.einsum('zjkbxv, jbv-> zjkbxv', 
                                    np.einsum('zqwbx, zjkqwv -> zjkbxv', 
                                    np.einsum('qwbxp, zp -> zqwbx', self.OmegaQ3, Control_zp3, optimize=opt), 
                                    Pesi, optimize=opt),  1/(1 + self.gammaBNU2 + 1j*self.DELTAEB2), optimize=opt)
                        )
            
            return Pge, Pes, Peb, Peb2
        elif self.protocol == 'ORCA_GSM':
            Sgs, E = coherences
            Control0 = Control[0]
            #M1 = Control[1]
            #M2 = Control[2]
            Control_zp0 = Control0(ti, self.zCheby)
            #M1_zp = M1(ti, self.zCheby)
            #M2_zp = M2(ti, self.zCheby)
            Pge1 = np.einsum('gjv, zghjkv -> zghjkv', 1/(1+1j*self.DELTAS),
                        (np.einsum('zghjkvp, zp -> zghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E, optimize=opt)          
                        -1j*np.einsum('zjkqw, zghqwv -> zghjkv', np.einsum('jkqwp, zp -> zjkqw', self.OmegaQ, Control_zp0, optimize=opt), Sgs, optimize=opt)
                        ) 
                    , optimize=opt)
            
            # Pge2 = np.einsum('gjv, zghjkv -> zghjkv', 1/(1+1j*self.DELTAGJ),
            #             (np.einsum('zghjkvp, zp -> zghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), ER, optimize=opt)          
            #             -1j*np.einsum('zjkqw, zghqwv -> zghjkv', np.einsum('jkqwp, zp -> zjkqw', self.OmegaQ, M1_zp, optimize=opt), Sgs, optimize=opt)
            #             -1j*np.einsum('zjkbx, zghbxv -> zghjkv', np.einsum('jkbxp, zp -> zjkbx', self.OmegaM2Q, M2_zp, optimize=opt), Sgb, optimize=opt)
            #             ) 
            #         , optimize=opt)
            return Pge1



    
    def Sderivative(self, ti, coherences, Control):
        opt=False
        if self.protocol == 'EIT' or self.protocol == 'FLAME':
            # (g, mg, j, mj, q, mq, v, z, Q)
            P, S = coherences
            dS = ( np.einsum('gqv, zghqwv -> zghqwv', -(self.gammaSNU + 1j*self.DELTA2), S, optimize=opt)
                  -1j* np.einsum('zjkqw, zghjkv -> zghqwv', np.einsum('jkqwp, zp -> zjkqw', self.OmegaQ, np.conj(Control(ti, self.zCheby)), optimize=opt), P, optimize=opt)
            )
            return dS
        elif self.protocol == 'EITFWM':
            # (g, mg, j, mj, q, mq, v, z, Q)
            P, S, EF = coherences
            Control_zp = Control(ti, self.zCheby)
            dS = ( np.einsum('gqv, zghqwv -> zghqwv', -(self.gammaSNU + 1j*self.DELTA2), S, optimize=opt)
                  -1j* np.einsum('zjkqw, zghjkv -> zghqwv', np.einsum('jkqwp, zp -> zjkqw', self.OmegaQ, np.conj(Control_zp), optimize=opt), P, optimize=opt)
                  - np.einsum('zghjk, zjkqwv -> zghqwv', np.einsum('ghjkp, zp -> zghjk', self.OmegaQF, Control_zp, optimize=opt), 
                                np.einsum('zghjkv, zghqwv -> zjkqwv', 
                                    np.einsum('zghjk, gjv -> zghjkv', 
                                                np.einsum('ghjkp, zp -> zghjk', self.OmegaQF, np.conj(Control_zp), optimize=opt), 1/(1-1j*self.DELTACHF), optimize=opt), S, optimize=opt), optimize=opt)
                +1j* np.einsum('zjkqwvp, zghjkvp -> zghqwv', self.dsqrtQF, 
                               np.einsum('zghjkv, zp -> zghjkvp', np.einsum('zghjk, gjv -> zghjkv', np.einsum('zghjk, gh -> zghjk', 
                                           np.einsum('ghjkp, zp -> zghjk', self.OmegaQF, np.conj(Control_zp), optimize=opt), np.sqrt(self.pop) , optimize=opt), 1/(1-1j*self.DELTACHF), optimize=opt), 
                                           np.conj(EF), optimize=opt), optimize=opt)
            )
            return dS
        elif self.protocol == 'RamanFWM' or self.protocol == 'RamanFWM-Magic':
            # (g, mg, j, mj, q, mq, v, z, Q)
            S, E, EF, Y, M = coherences
            Control_zp = Control(ti, self.zCheby)
            dS = ( np.einsum('gqv, zghqwv -> zghqwv', -(self.gammaSNU + 1j*self.DELTA2), S, optimize=opt)

                   - np.einsum('jkqwz, ghjkvz -> zghqwv', np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                                     np.einsum('ghjkvz, gjv -> ghjkvz', np.einsum('jkqwz, zghqwv -> ghjkvz' ,
                                                                                                  np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  Control_zp, optimize=opt), S, optimize=opt),
                                                               (1/(1 + 1j*self.DELTAS)), optimize=opt), optimize=opt)

                    - 1j* np.einsum('jkqwz, ghjkvz -> zghqwv', np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('ghjkvz, gjv -> ghjkvz' , 
                                    np.einsum('zghjkvp, zp -> ghjkvz', 
                                                np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS)), optimize=opt))

                  - np.einsum('zghjk, zjkqwv -> zghqwv', np.einsum('ghjkp, zp -> zghjk', self.OmegaQF, Control_zp, optimize=opt), 
                                np.einsum('zghjkv, zghqwv -> zjkqwv', 
                                    np.einsum('zghjk, gjv -> zghjkv', 
                                                np.einsum('ghjkp, zp -> zghjk', self.OmegaQF, np.conj(Control_zp), optimize=opt), 1/(1-1j*self.DELTACHF), optimize=opt), S, optimize=opt), optimize=opt)

                +1j* np.einsum('zjkqwvp, zghjkvp -> zghqwv', self.dsqrtQF, 
                               np.einsum('zghjkv, zp -> zghjkvp', np.einsum('zghjk, gjv -> zghjkv', np.einsum('zghjk, gh -> zghjk', 
                                           np.einsum('ghjkp, zp -> zghjk', self.OmegaQF, np.conj(Control_zp), optimize=opt), np.sqrt(self.pop) , optimize=opt), 1/(1+1j*self.DELTACHF), optimize=opt), 
                                           np.conj(EF), optimize=opt), optimize=opt)

                + 1j* np.einsum('jkqwz, ghjkvz -> zghqwv', np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('zjkbxv, ghbxvz -> ghjkvz' , np.conj(Y),
                                    np.einsum('ghbxvz, gbv -> ghbxvz' , 
                                    np.einsum('zghbxvp, zp -> ghbxvz', 
                                                np.einsum('zghbxvp, gh -> zghbxvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS))), optimize=opt))
                
                - 1j* np.einsum('jkqwz, ghjkvz -> zghqwv', np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('zghasv, asjkvz -> ghjkvz' , M,
                                    np.einsum('asjkvz, ajv -> asjkvz' , 
                                    np.einsum('zasjkvp, zp -> asjkvz', 
                                                np.einsum('zasjkvp, as -> zasjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS))), optimize=opt))

                - 1j* np.einsum('ghjkz, qwjkvz -> zghqwv', np.einsum('ghjkp, zp -> ghjkz', self.OmegaQF,  Control_zp, optimize=opt), 
                                    np.einsum('zjkbxv, qwbxvz -> qwjkvz' , Y,
                                    np.einsum('qwbxvz, qbv -> qwbxvz' , 
                                    np.einsum('zqwbxvp, zp -> qwbxvz', 
                                                self.dsqrtQF, np.conj(EF)), (1/(1 - 1j*self.DELTACHF))), optimize=opt))
                
            )
            return dS
        elif self.protocol == 'Raman' or self.protocol == 'Raman_test' or self.protocol=='ORCA':
            # (g, mg, j, mj, q, mq, v, z, Q)
            S, E = coherences
            Control_zp = Control(ti, self.zCheby)
            dS = (
                    np.einsum('gqv,zghqwv -> zghqwv', 
                                        -(self.gammaSNU + 1j*self.DELTA2)
                                        , S, optimize=opt)

                    - np.einsum('jkqwz, ghjkvz -> zghqwv', np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                np.einsum('ghjkvz, gjv -> ghjkvz', np.einsum('jkqwz, zghqwv -> ghjkvz' ,np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  Control_zp, optimize=opt), S, optimize=opt),
                                        (1/(1 + 1j*self.DELTAS)), optimize=opt), optimize=opt)

                    - 1j* np.einsum('zjkqw, zghjkv -> zghqwv', np.einsum('jkqwp,zp -> zjkqw', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('zghjkv, gjv -> zghjkv' , 
                                    np.einsum('zghjkvp, zp -> zghjkv', 
                                                np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS)), optimize=opt))
                                            
                                            )
            return dS
        elif self.protocol == 'TORCA':
            #(g, mg, j, mj, q, mq, v, z, Q)
            S, E = coherences
            Control_zp = Control(ti, self.zCheby)
            dS = (
                np.einsum('gqv, zghqwv -> zghqwv', -(self.gammaSNU + 1j*self.DELTA2), S)

                - np.einsum('ghjkz, jkqwvz -> zghqwv', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                                     np.einsum('jkqwvz, jqv -> jkqwvz', 
                                                               np.einsum('ghjkz, zghqwv -> jkqwvz' ,
                                                                         np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  Control_zp, optimize=opt), S, optimize=opt),
                                                               (1/(1 + self.gammaSNU + 1j*self.DELTAS)), optimize=opt), optimize=opt)

                -1j* np.einsum('zjkqwvp, ghjkvzp -> zghqwv', self.dsqrtQ,
                               np.einsum('ghjkz, gjvzp -> ghjkvzp', 
                               np.einsum('ghjkz, gh -> ghjkz', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  Control_zp, optimize=opt), np.sqrt(self.pop)),
                               np.einsum('zp, gjv -> gjvzp', E, 1/(1 + 1j*self.DELTAC)))
                            )
                )
            return dS
        elif self.protocol == 'TORCAP':
            # (g, mg, j, mj, q, mq, v, z, Q)
            Pge, Pes, S, E = coherences
            Control_zp = Control(ti, self.zCheby)
            dS = (
                    np.einsum('gqv,zghqwv->zghqwv', 
                                        -(self.gammaSNU + 1j*self.DELTA2)
                                        , S, optimize=opt)

                    + 1j* np.einsum('ghjkz,zjkqwv->zghqwv', 
                    np.einsum('ghjkp,zp->ghjkz', self.OmegaQ,  Control_zp), Pes, optimize=opt)

                    + np.einsum('zghjkv, jkqwvz -> zghqwv', Pge, np.einsum('zjkqwvp, zp -> jkqwvz', self.dsqrtQ, E) )
                        )
            return dS
        elif self.protocol == '4levelTORCAP':
            #(g, mg, j, mj, q, mq, v, z, Q)
            Pge, Pes, Peb, Sgs, Sgb, E = coherences
            Control1 = Control[0]
            Control2 = Control[1]
            Control_zp1 = Control1(ti, self.zCheby)
            Control_zp2 = Control2(ti, self.zCheby)

            dSgs = np.einsum('ghqwvz->zghqwv', (
                                            np.einsum('gqv,zghqwv -> ghqwvz', 
                                                                -(self.gammaSNU + 1j*self.DELTAGS)
                                                                , Sgs, optimize=opt)

                                            + 1j* np.einsum('ghjkz,zjkqwv -> ghqwvz', 
                                            np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  Control_zp1), Pes, optimize=opt)

                                            + np.einsum('zghjkv, jkqwvz -> ghqwvz', Pge, np.einsum('zjkqwvp, zp -> jkqwvz', self.dsqrtQ, E) )

                                            - 1j* np.einsum('qwbxz,zghbxv -> ghqwvz', 
                                            np.einsum('qwbxp,zp -> qwbxz', self.OmegaQ2,  np.conj(Control_zp2)), Sgb, optimize=opt)
                                            )
                        )
            
            # dSgs = np.einsum('ghqwvz->zghqwv', (
            #                                 np.einsum('gqv,zghqwv -> ghqwvz', 
            #                                                     -(self.gammaSNU + 1j*self.DELTAGS)
            #                                                     , Sgs, optimize=opt)

            #                                 + 1j* np.einsum('ghjkz,zjkqwv -> ghqwvz', 
            #                                 np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  Control_zp1), Pes, optimize=opt)

            #                                 - 1j* np.einsum('zghjkv, jkqwvz -> ghqwvz', 
            #                                 np.einsum('ghjkz, gjv -> zghjkv' , 
            #                                 np.einsum('ghjkz, gh -> ghjkz', 
            #                                 np.einsum('ghjkp, zp -> ghjkz', self.OmegaQ,  Control_zp1),  np.sqrt(self.pop)), 1/(1+1j*self.DELTAC)), 
            #                                 np.einsum('zjkqwvp, zp -> jkqwvz', self.dsqrtQ, E) )

            #                                 - 1j* np.einsum('qwbxz,zghbxv -> ghqwvz', 
            #                                 np.einsum('qwbxp,zp -> qwbxz', self.OmegaQ2,  np.conj(Control_zp2)), Sgb, optimize=opt)
            #                                 )
            #             )

            dSgb = np.einsum('ghbxvz->zghbxv', (
                                            np.einsum('gbv,zghbxv->ghbxvz', 
                                                                -(self.gammaBNU + 1j*self.DELTAGB)
                                                                , Sgb, optimize=opt)

                                            + 1j* np.einsum('ghjkz,zjkbxv->ghbxvz', 
                                            np.einsum('ghjkp,zp->ghjkz', self.OmegaQ,  Control_zp1), Peb, optimize=opt)

                                            - 1j* np.einsum('qwbxz,zghqwv->ghbxvz', 
                                            np.einsum('qwbxp,zp->qwbxz', self.OmegaQ2,  Control_zp2), Sgs, optimize=opt)
                                            )
                        )
            return dSgs, dSgb
        elif self.protocol == 'TORCAP_2dressing_states':
            #(g, mg, j, mj, q, mq, v, z, Q)
            Pge, Pes, Peb, Peb2, Sgs, Sgb, Sgb2, E = coherences
            Control1 = Control[0]
            Control2 = Control[1]
            Control_zp1 = Control1(ti, self.zCheby)
            Control_zp2 = Control2(ti, self.zCheby)
            Control_zp3 = self.rabi_modification*Control_zp2

            dSgs = np.einsum('ghqwvz->zghqwv', (
                                            np.einsum('gqv,zghqwv -> ghqwvz', 
                                                                -(self.gammaSNU + 1j*self.DELTAGS)
                                                                , Sgs, optimize=opt)

                                            + 1j* np.einsum('ghjkz,zjkqwv -> ghqwvz', 
                                            np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  Control_zp1), Pes, optimize=opt)

                                            + np.einsum('zghjkv, jkqwvz -> ghqwvz', Pge, np.einsum('zjkqwvp, zp -> jkqwvz', self.dsqrtQ, E) )

                                            - 1j* np.einsum('qwbxz,zghbxv -> ghqwvz', 
                                            np.einsum('qwbxp,zp -> qwbxz', self.OmegaQ2,  np.conj(Control_zp2)), Sgb, optimize=opt)

                                            - 1j* np.einsum('qwbxz,zghbxv -> ghqwvz', 
                                            np.einsum('qwbxp,zp -> qwbxz', self.OmegaQ3,  np.conj(Control_zp3)), Sgb2, optimize=opt)
                                            )
                        )

            dSgb = np.einsum('ghbxvz->zghbxv', (
                                            np.einsum('gbv,zghbxv->ghbxvz', 
                                                                -(self.gammaBNU + 1j*self.DELTAGB)
                                                                , Sgb, optimize=opt)

                                            + 1j* np.einsum('ghjkz,zjkbxv->ghbxvz', 
                                            np.einsum('ghjkp,zp->ghjkz', self.OmegaQ,  Control_zp1), Peb, optimize=opt)

                                            - 1j* np.einsum('qwbxz,zghqwv->ghbxvz', 
                                            np.einsum('qwbxp,zp->qwbxz', self.OmegaQ2,  Control_zp2), Sgs, optimize=opt)
                                            )
                        )
            
            dSgb2 = np.einsum('ghbxvz->zghbxv', (
                                            np.einsum('gbv,zghbxv->ghbxvz', 
                                                                -(self.gammaBNU2 + 1j*self.DELTAGB2)
                                                                , Sgb2, optimize=opt)

                                            + 1j* np.einsum('ghjkz,zjkbxv->ghbxvz', 
                                            np.einsum('ghjkp,zp->ghjkz', self.OmegaQ,  Control_zp1), Peb2, optimize=opt)

                                            - 1j* np.einsum('qwbxz,zghqwv->ghbxvz', 
                                            np.einsum('qwbxp,zp->qwbxz', self.OmegaQ3,  Control_zp3), Sgs, optimize=opt)
                                            )
                        )
            return dSgs, dSgb, dSgb2
        elif self.protocol == 'ORCA_GSM':
            # (g, mg, j, mj, q, mq, v, z, Q)
            Pge1, Pge2, Sgs, Sgb = coherences
            Control0 = Control[0]
            M1 = Control[1]
            M2 = Control[2]
            Control_zp0 = Control0(ti, self.zCheby)
            M1_zp = M1(ti, self.zCheby)
            M2_zp = M2(ti, self.zCheby)

            dSgs = ( np.einsum('gqv, zghqwv -> zghqwv', -(self.gammaSNU + 1j*self.DELTAGS), Sgs, optimize=opt)
                  -1j* np.einsum('zjkqw, zghjkv -> zghqwv', np.einsum('jkqwp, zp -> zjkqw', self.OmegaQ, np.conj(Control_zp0), optimize=opt), Pge1, optimize=opt)
                  -1j* np.einsum('zjkqw, zghjkv -> zghqwv', np.einsum('jkqwp, zp -> zjkqw', self.OmegaQ, np.conj(M1_zp), optimize=opt), Pge2, optimize=opt)
            )
            dSgb = ( np.einsum('gbv, zghbxv -> zghbxv', -(self.gammaBNU + 1j*self.DELTAGB), Sgb, optimize=opt)
                  -1j* np.einsum('zjkbx, zghjkv -> zghbxv', np.einsum('bxjkp, zp -> zjkbx', self.OmegaM2Q, np.conj(M2_zp), optimize=opt), Pge2, optimize=opt)
            )
            return dSgs, dSgb
        
        elif self.protocol == 'TORCA_GSM_D1' or self.protocol == 'TORCA_GSM_D2':
            P2, S, S2, E = coherences
            [Control1, Control2, Control3] = Control 
            Control_zp1 = Control1(ti, self.zCheby)
            Control_zp2 = Control2(ti, self.zCheby)
            Control_zp3 = Control3(ti, self.zCheby)

            dS = (
                np.einsum('gqv, zghqwv -> zghqwv', -(self.gammaSNU + 1j*self.DELTAGS), S)

                - np.einsum('ghjkz, jkqwvz -> zghqwv', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  np.conj(Control_zp1), optimize=opt), 
                                                     np.einsum('jkqwvz, jqv -> jkqwvz', 
                                                               np.einsum('ghjkz, zghqwv -> jkqwvz' ,
                                                                         np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  Control_zp1, optimize=opt), S, optimize=opt),
                                                               (1/(1 + self.gammaSNU + 1j*self.DELTAS)), optimize=opt), optimize=opt)

                -1j* np.einsum('zjkqwvp, ghjkvzp -> zghqwv', self.dsqrtQ,
                               np.einsum('ghjkz, gjvzp -> ghjkvzp', 
                               np.einsum('ghjkz, gh -> ghjkz', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  Control_zp, optimize=opt), np.sqrt(self.pop)),
                               np.einsum('zp, gjv -> gjvzp', E, 1/(1 + 1j*self.DELTAC)))
                            )

                - 1j* np.einsum('qwbxz,zghbxv -> ghqwvz', 
                                            np.einsum('qwbxp,zp -> qwbxz', self.OmegaQ2,  np.conj(Control_zp2)), P2, optimize=opt)

                )
            
            dS2 = (
                                            np.einsum('gav,zghayv->zghayv', 
                                                                -(self.gammaSNU2 + 1j*self.DELTAGS2)
                                                                , S2, optimize=opt)

                                            - 1j* np.einsum('bxayz,zghbxv->zghayv', 
                                            np.einsum('bxayp,zp->bxayz', self.OmegaQ3,  np.conj(Control_zp3)), P2, optimize=opt)

                    )
            return dS, dS2
        
    def coherence_corr(self, coherence):
        """ Makes a boolean array to be used with Y and M coherences. 
            This is to ensure a state can't have a coherence with itself."""
        
        Flength = len(coherence[0, 0, :, 0, 0, 0, 0]) # number of F states
        Findices = np.array(np.where(np.eye(Flength))).T
        mFlength = len(coherence[0, 0, 0, :, 0, 0, 0])
        mFindices = np.array(np.where(np.eye(mFlength))).T
        indices = np.concatenate((np.array(list(itertools.product(Findices[:, 0], mFindices[:, 0]))), np.array(list(itertools.product(Findices[:, 0], mFindices[:, 0])))), axis=1).reshape(-1, 4)
        arr = np.zeros((Flength, mFlength, Flength, mFlength), dtype=bool)
        arr[tuple(indices.T)] = True
        return arr
    
    def Yderivative(self, ti, coherences, Control):
        opt = False
        if self.protocol == 'RamanFWM' or self.protocol == 'RamanFWM-Magic':
            S, E, Y, M = coherences
            Control_zp = Control(ti, self.zCheby)
            dY = (
                    np.einsum('jb,zjkbxv->zjkbxv', 
                                        -(2 + 1j*self.chi)
                                        , Y, optimize=opt)
                    -2*np.real(
                        np.einsum('ghjkz, ghbxvz -> zjkbxv', np.einsum('ghjkp, zp -> ghjkz', self.OmegaQF,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('ghbxvz, gbv -> ghbxvz' , 
                                    np.einsum('zghbxvp, zp -> ghbxvz', 
                                                np.einsum('zghbxvp, gh -> zghbxvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS)), optimize=opt))

                        + np.einsum('ghjkz, ghbxvz -> zjkbxv', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQF,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('zjkbxv, ghjkvz -> ghbxvz' , np.conj(Y),
                                    np.einsum('ghjkvz, gjv -> ghjkvz' , 
                                    np.einsum('zghjkvp, zp -> ghjkvz', 
                                                np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS))), optimize=opt))

                        + np.einsum('ghjkz, ghbxvz -> zjkbxv', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('zghasv, asbxvz -> ghbxvz' , M,
                                    np.einsum('asbxvz, abv -> asbxvz' , 
                                    np.einsum('zasbxvp, zp -> asbxvz', 
                                                np.einsum('zasbxvp, as -> zasbxvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS))), optimize=opt))

                        - 1j*np.einsum('ghjkz, ghbxvz -> zjkbxv', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQF,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('ghbxvz, gbv -> ghbxvz', np.einsum('bxqwz, zghqwv -> ghbxvz' ,np.einsum('bxqwp,zp -> bxqwz', self.OmegaQ,  Control_zp, optimize=opt), S, optimize=opt),
                                            (1/(1 + 1j*self.DELTAS)), optimize=opt), optimize=opt)

                    )
            )
        elif self.protocol == 'TORCAP':
            Pge, Pes, Y, E = coherences
            Control_zp = Control(ti, self.zCheby)
            dY = (
                    np.einsum('jb,zjkbxv->zjkbxv', 
                                        -(2 + 1j*self.chi)
                                        , Y, optimize=opt)

                    + 1j* np.einsum('ghjkz,zghbxv->zjkbxv',

                        np.einsum('ghjkp,zp->ghjkz', self.OmegaQ,  np.conj(Control_zp)), Pge, optimize=opt
                        
                        )

                    - 1j* np.einsum('ghbxz,zghjkv->zjkbxv',

                        np.einsum('ghbxp,zp->ghbxz', self.OmegaQ,  Control_zp), np.conj(Pge), optimize=opt
                        
                        )
                    
                    )
            # level can't have coherences with itself, so multiply by tensor with zeros along j=b?
            # seems hacky?


        elif self.protocol == '4levelTORCAP' or self.protocol == 'TORCAP_2dressing_states':
            Pge, Pes, Y, E = coherences
            Control1 = Control[0]
            Control_zp1 = Control1(ti, self.zCheby)
            dY = (
                    np.einsum('jb,zjkbxv->zjkbxv', 
                                        -(2 + 1j*self.chi)
                                        , Y, optimize=opt)

                    + 1j* np.einsum('ghjkz,zghbxv->zjkbxv',

                        np.einsum('ghjkp,zp->ghjkz', self.OmegaQ,  np.conj(Control_zp1)), Pge, optimize=opt
                        
                        )

                    - 1j* np.einsum('ghbxz,zghjkv->zjkbxv',

                        np.einsum('ghbxp,zp->ghbxz', self.OmegaQ,  Control_zp1), np.conj(Pge), optimize=opt
                        
                        )
                    
                    )
        dY[:, self.Ycorrection, :] = 0
        return dY
        
    def Mderivative(self, ti, coherences, Control):
        opt = False
        if self.protocol == 'RamanFWM' or self.protocol == 'RamanFWM-Magic':
            S, E, Y, M = coherences
            Control_zp = Control(ti, self.zCheby)
            dM = (
                    -2*np.real(
                        np.einsum('ghjkz, asjkvz -> zghasv', np.einsum('ghjkp, zp -> ghjkz', self.OmegaQF,  Control_zp, optimize=opt), 
                                    np.einsum('asjkvz, gjv -> asjkvz' , 
                                    np.einsum('zasjkvp, zp -> asjkvz', 
                                                np.einsum('zasjkvp, as -> zasjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS)), optimize=opt))

                        + np.einsum('ghjkz, asjkvz -> zghasv', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQF,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('zjkbxv, asjkvz -> asjkvz' , np.conj(Y),
                                    np.einsum('asjkvz, ajv -> asjkvz' , 
                                    np.einsum('zasjkvp, zp -> asjkvz', 
                                                np.einsum('zasjkvp, as -> zasjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS))), optimize=opt))

                        + np.einsum('ghjkz, asjkvz -> zghasv', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQ,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('zghasv, ghjkvz -> asjkvz' , M,
                                    np.einsum('ghjkvz, gjv -> ghjkvz' , 
                                    np.einsum('zghjkvp, zp -> ghjkvz', 
                                                np.einsum('zghjkvp, gh -> zghjkvp', self.dsqrtQ, np.sqrt(self.pop), optimize=opt), E), (1/(1 + 1j*self.DELTAS))), optimize=opt))

                        - 1j*np.einsum('ghjkz, asjkvz -> zghasv', np.einsum('ghjkp,zp -> ghjkz', self.OmegaQF,  np.conj(Control_zp), optimize=opt), 
                                    np.einsum('asjkvz, ajv -> asjkvz', np.einsum('jkqwz, zasqwv -> asjkvz' ,np.einsum('jkqwp,zp -> jkqwz', self.OmegaQ,  Control_zp, optimize=opt), S, optimize=opt),
                                            (1/(1 + 1j*self.DELTAS)), optimize=opt), optimize=opt)

                    )
            )
        elif self.protocol == 'TORCAP':
            Pge = coherences
            Control_zp = Control(ti, self.zCheby)
            dM = (

                    + 1j* np.einsum('ghjkz,zasjkv->zghasv',

                        np.einsum('ghjkp,zp->ghjkz', self.OmegaQ,  Control_zp), np.conj(Pge), optimize=opt
                        
                        )

                    - 1j* np.einsum('asjkz,zghjkv->zghasv',

                        np.einsum('asjkp,zp->asjkz', self.OmegaQ,  np.conj(Control_zp)),Pge, optimize=opt
                        
                        )
                    
                    )
        elif self.protocol == '4levelTORCAP' or self.protocol == 'TORCAP_2dressing_states':
            Pge = coherences
            Control1 = Control[0]
            Control_zp1 = Control1(ti, self.zCheby)
            dM = (

                    + 1j* np.einsum('ghjkz,zasjkv->zghasv',

                        np.einsum('ghjkp,zp->ghjkz', self.OmegaQ,  Control_zp1), np.conj(Pge), optimize=opt
                        
                        )

                    - 1j* np.einsum('asjkz,zghjkv->zghasv',

                        np.einsum('asjkp,zp->asjkz', self.OmegaQ,  np.conj(Control_zp1)),Pge, optimize=opt
                        
                        )
                    
                    )
        dM[:, self.Mcorrection, :] = 0
        return dM

        
    def storage_efficiency(self, S, mi):
        # if self.vno == 1:
        #     return ( np.trapz(
        #     np.sum(
        #         np.sum(
        #             np.sum(
        #                 np.sum(
        #                     np.sum(
        #                         pow(np.nan_to_num(np.abs(S[mi, :, :, :, :, :, :]), nan=0.0), 2),
        #                         axis=-1), 
        #                     axis=-1), 
        #                 axis=-1), 
        #             axis=-1), 
        #         axis=-1), self.zCheby)
        #     )
        # else:
        return ( np.trapz( 
                    np.einsum('zghqw -> z', 
                                pow(np.abs(np.einsum('v, zghqwv -> zghqw', np.sqrt(self.MB(self.vs)*self.dvs), np.nan_to_num(S[mi], nan=0.0))), 2)), 
                                self.zCheby)
        )

    
    def retrieval_efficiency(self, E, mi, p):
        return ( np.trapz(pow(np.abs(E[mi:, -1, p]), 2), self.tpoints[mi:]) )
        
    def check_energy(self):
        total = 0
        for coherence in self.coherences_list:
            total += self.storage_efficiency(coherence, -1)

        total += self.retrieval_efficiency(self.E, 0, 0) + self.retrieval_efficiency(self.E, 0, 1)

        return total
        
        

def photon_gaussian(t, t0, FWHM):
    """
    Gaussian function.  Returns value at specified time.
    t0 = peak time
    FWHM = full width at half maximum
    """
    A = pow(8*np.log(2)/(np.power(FWHM,2)*np.pi), 0.25)
    return A*np.exp(-4*np.log(2)*np.power(t - t0, 2.) / (np.power(FWHM, 2.)))

def photon_decaying_exp(t, t0, tau):
    A = np.sqrt(2/tau)
    output = A*np.exp(-(t-t0)/tau)
    output[t<t0] = 0
    return output


def gaussian(t, t0, FWHM, A):
    """
    Gaussian function.  Returns value at specified time.
    t0 = peak time
    FWHM = full width at half maximum
    """
    return A*np.exp(-4*np.log(2)*np.power(t - t0, 2.) / (np.power(FWHM, 2.))) 