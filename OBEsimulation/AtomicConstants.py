import numpy as np
import pandas as pd
from scipy.constants import physical_constants, epsilon_0, hbar, c, e, h
from scipy.linalg import kron, eigh
from sympy.physics.quantum.cg import Wigner6j
from sympy.physics.quantum.cg import CG
from fractions import Fraction
amu = physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
au = physical_constants['atomic unit of electric dipole mom.'][0] # atomic units for reduced dipole moment

class Alkalis:
    def __init__(self):
        self.S = 1/2
        self.Lg = 0
        self.Jg = 1/2
        self.Lj = 1

        self.unpack_config()

        self.lookup_atom_constants()

        # hyperfine splitting - calculate?
        self.hyperfine_splitting()

        # coupling tensor - separate ones for ge, es, etc?
        self.coupling_ge = self.coupling(self.Jg, self.Fg, self.mg, self.Jj, self.Fj, self.mj)
        self.coupling_es = self.coupling(self.Jj, self.Fj, self.mj, self.Jq, self.Fq, self.mq)
        # dressing
        if self.number_of_states == 4:
            self.coupling_sb = self.coupling(self.Jq, self.Fq, self.mq, self.Jb, self.Fb, self.mb)
        if self.number_of_states == 5: # ground state mapping or dressing with two states
            # dressing with two states
            if 'dressing2' in self.config["states"].keys():
                self.coupling_sb = self.coupling(self.Jq, self.Fq, self.mq, self.Jb, self.Fb, self.mb) # assuming 'dresssing' exists if 'dressing2' exists
                self.coupling_sb2 = self.coupling(self.Jq, self.Fq, self.mq, self.Jb2, self.Fb2, self.mb2)
            # ground state mapping
            else:
                self.coupling_be = self.coupling(self.Jb, self.Fb, self.mb, self.Jj, self.Fj, self.mj)

    def __repr__(self):
        ground_state = self.ground_state.replace('/', 'q') # use q instead of / for use in file names
        intermediate_state = self.intermediate_state.replace('/', 'q')
        storage_state = self.storage_state.replace('/', 'q')
        if self.config["states"]["storage"]["L"] == 0: # using ground state to store i.e. lambda scheme
            string = f"{self.atom}_{ground_state}F{self.Fg}_{intermediate_state}_{storage_state}F{self.Fq}"
        else:
            string = f"{self.atom}_{ground_state}F{self.Fg}_{intermediate_state}_{storage_state}"
        # dressing state
        if self.number_of_states == 4:
            string += f"_{self.dressing_state.replace('/', 'q')}"

        # ground state mapping
        if self.number_of_states == 5:
            # dressing with two states
            if 'dressing2' in self.config["states"].keys():
                string += f"_{self.dressing_state.replace('/', 'q')}"
                string += f"_{self.dressing_state2.replace('/', 'q')}"
            else:
                string += f"_{intermediate_state}_{ground_state}F{self.Fb}"

        if self.splitting:
            string += f"_HyperfineSplittingYes"
        else:
            string += f"_HyperfineSplittingNo"
        
        return string 

    
    def hyperfine_states(self, I, J):
        Fmin = np.abs(J - I)
        Fmax = np.abs(J + I)
        F = np.arange(Fmin, Fmax+1)
        return F
    
    def momentum_letter(self, L):
        if L == 0:
            return 's'
        elif L == 1:
            return 'p'
        elif L == 2:
            return 'd'
        elif L == 3:
            return 'f'
    
    def unpack_config(self):
        
        self.number_of_states = len(self.config["states"])
        self.splitting = self.config["Hyperfine splitting"]

        self.nj = self.config["states"]["intermediate"]["n"]
        self.Jj = self.config["states"]["intermediate"]["J"]

        if self.config["states"]["storage"]["L"] == 0: # using ground state to store i.e. lambda scheme
            self.nq = self.ng
            self.Jq = self.Jg
            self.Lq = self.Lg
        else:
            self.nq = self.config["states"]["storage"]["n"]
            self.Jq = self.config["states"]["storage"]["J"]
            self.Lq = self.config["states"]["storage"]["L"]

        # dressing state
        if self.number_of_states == 4:
            self.nb = self.config["states"]["dressing"]["n"]
            self.Lb = self.config["states"]["dressing"]["L"]
            self.Jb = self.config["states"]["dressing"]["J"]

        if self.number_of_states == 5: # ground state mapping or dressing with two states
            if 'dressing2' in self.config["states"].keys(): # dressing with two states
                self.nb = self.config["states"]["dressing"]["n"]
                self.Lb = self.config["states"]["dressing"]["L"]
                self.Jb = self.config["states"]["dressing"]["J"]
                self.nb2 = self.config["states"]["dressing2"]["n"]
                self.Lb2 = self.config["states"]["dressing2"]["L"]
                self.Jb2 = self.config["states"]["dressing2"]["J"]

            else: # ground state mapping
                # assume intermediate excited state the same for mapping up to mapping down
                self.nb = self.ng #self.config["states"]["dressing"]["n"]
                self.Lb = self.Lg #self.config["states"]["dressing"]["L"]
                self.Jb = self.Jg #self.config["states"]["dressing"]["J"]


        if self.splitting == False:
            self.Fg = np.array([0])
            self.mg = np.array([0])

            self.Fj = np.array([0])
            self.mj = np.array([0])

            self.Fq = np.array([0])
            self.mq = np.array([0])

            # dressing state
            if self.number_of_states == 4:
                self.Fb = np.array([0])
                self.mb = np.array([0])         

            # ground state mapping 
            if self.number_of_states == 5:
                if 'dressing2' in self.config["states"].keys(): # dressing with two states
                    self.Fb = np.array([0])
                    self.mb = np.array([0])         
                    self.Fb2 = np.array([0])
                    self.mb2 = np.array([0])  
                else: # ground state mapping
                    self.Fb = np.array([0])
                    self.mb = np.array([0]) 

        else: # hyperfine splitting
            # Ground state
            self.Fg = np.array([self.config["states"]["initial"]["F"]])
            self.mg = np.arange(-max(self.Fg), max(self.Fg)+1)

            # intermediate state
            self.Fj = self.hyperfine_states(self.I, self.Jj)
            self.mj = np.arange(-max(self.Fj), max(self.Fj)+1)

            # storage state
            if self.config["states"]["storage"]["L"] == 0: # using ground state to store i.e. lambda scheme
                # could have same F as initial state, need to be pumped into initial mF state!
                self.Fq = np.array([self.config["states"]["storage"]["F"]])
                self.mq = np.arange(-max(self.Fq), max(self.Fq)+1)
            else:
                self.Fq = self.hyperfine_states(self.I, self.Jq)
                self.mq = np.arange(-max(self.Fq), max(self.Fq)+1)

            # dressing state
            if self.number_of_states == 4:
                self.Fb = self.hyperfine_states(self.I, self.Jb)
                self.mb = np.arange(-max(self.Fb), max(self.Fb)+1)

            # ground state mapping
            if self.number_of_states == 5:
                if 'dressing2' in self.config["states"].keys(): # dressing with two states
                    self.Fb = self.hyperfine_states(self.I, self.Jb)
                    self.mb = np.arange(-max(self.Fb), max(self.Fb)+1)
                    self.Fb2 = self.hyperfine_states(self.I, self.Jb2)
                    self.mb2 = np.arange(-max(self.Fb2), max(self.Fb2)+1)
                else: # ground state mapping
                    self.Fb = np.array([self.config["states"]["storage2"]["F"]])
                    self.mb = np.arange(-max(self.Fb), max(self.Fb)+1)

    def lookup_atom_constants(self):
        self.ground_state = str(self.ng)+self.momentum_letter(self.Lg)+str(Fraction(self.Jg))
        self.intermediate_state = str(self.nj)+self.momentum_letter(self.Lj)+str(Fraction(self.Jj))
        self.storage_state = str(self.nq)+self.momentum_letter(self.Lq)+str(Fraction(self.Jq))

        df = pd.read_csv(self.filename)

        wavelengths = np.zeros(self.number_of_states-1)
        reduced_dipoles = np.zeros(self.number_of_states-1)
        self.reduced_dipoles = np.zeros(self.number_of_states - 1)
        self.lifetimes = np.zeros(self.number_of_states - 1)

        self.J_array = np.zeros(self.number_of_states)
        self.J_array[0] = self.Jg
        self.J_array[1] = self.Jj
        self.J_array[2] = self.Jq

        # make into loop?
        query = df[
                        (df['Initial'].str.contains(self.ground_state) & df['Final'].str.contains(self.intermediate_state)) |
                        (df['Final'].str.contains(self.ground_state) & df['Initial'].str.contains(self.intermediate_state))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

        [[wavelengths[0], reduced_dipoles[0]]] = query

        query = df[
                        (df['Initial'].str.contains(self.intermediate_state) & df['Final'].str.contains(self.storage_state)) |
                        (df['Final'].str.contains(self.intermediate_state) & df['Initial'].str.contains(self.storage_state))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

        [[wavelengths[1], reduced_dipoles[1]]] = query

        self.reduced_dipoles[0] = reduced_dipoles[0]*au/np.sqrt(2*self.J_array[0]+1) # to make same convention as Steck
        self.reduced_dipoles[1] = reduced_dipoles[1]*au/np.sqrt(2*self.J_array[1]+1)

        if self.number_of_states == 4:
            # dressing
            self.dressing_state = str(self.nb)+self.momentum_letter(self.Lb)+str(Fraction(self.Jb))
            query = df[
                        (df['Initial'].str.contains(self.storage_state) & df['Final'].str.contains(self.dressing_state)) |
                        (df['Final'].str.contains(self.storage_state) & df['Initial'].str.contains(self.dressing_state))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

            [[wavelengths[2], reduced_dipoles[2]]] = query
            self.reduced_dipoles[2] = reduced_dipoles[2]*au/np.sqrt(2*self.J_array[2]+1)
            self.J_array[3] = self.Jb


        elif self.number_of_states == 5:
            if 'dressing2' in self.config["states"].keys(): # dressing with two states
                # dressing
                self.dressing_state = str(self.nb)+self.momentum_letter(self.Lb)+str(Fraction(self.Jb))
                query = df[
                            (df['Initial'].str.contains(self.storage_state) & df['Final'].str.contains(self.dressing_state)) |
                            (df['Final'].str.contains(self.storage_state) & df['Initial'].str.contains(self.dressing_state))
                        ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

                [[wavelengths[2], reduced_dipoles[2]]] = query
                self.reduced_dipoles[2] = reduced_dipoles[2]*au/np.sqrt(2*self.J_array[2]+1)
                self.J_array[3] = self.Jb
                # dressing 2
                self.dressing_state2 = str(self.nb2)+self.momentum_letter(self.Lb2)+str(Fraction(self.Jb2))
                query = df[
                        (df['Initial'].str.contains(self.storage_state) & df['Final'].str.contains(self.dressing_state2)) |
                        (df['Final'].str.contains(self.storage_state) & df['Initial'].str.contains(self.dressing_state2))
                    ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()
                [[wavelengths[3], reduced_dipoles[3]]] = query
                self.reduced_dipoles[3] = reduced_dipoles[3]*au/np.sqrt(2*self.J_array[3]+1)
                self.J_array[4] = self.Jb2
            else:
                # ground state mapping
                # mapping field 1 has the same wavelength and dipole as control field (for ORCA)
                [wavelengths[2], reduced_dipoles[2]] = [wavelengths[1], reduced_dipoles[1]]
                self.reduced_dipoles[2] = reduced_dipoles[2]*au/np.sqrt(2*self.J_array[1]+1) # (j -> q)
                self.J_array[3] = self.Jj
                # mapping field 2 has the same wavelength and dipole as signal field (for ORCA)
                [wavelengths[3], reduced_dipoles[3]] = [wavelengths[0], reduced_dipoles[0]]
                self.reduced_dipoles[3] = reduced_dipoles[3]*au/np.sqrt(2*self.J_array[0]+1) # (g -> j)
                self.J_array[4] = self.Jg
            

        self.wavelengths = wavelengths*1e-9 # convert to m
        self.angular_frequencies = 2*np.pi*c/self.wavelengths
        self.wavevectors = 2*np.pi/self.wavelengths

        self.lifetimes = ( 3*np.pi*epsilon_0*hbar*pow(c,3) * 
             (2*self.J_array[1:] + 1)/(pow(self.angular_frequencies,3)*(2*self.J_array[:-1] + 1)*pow(self.reduced_dipoles,2)) )
        
        if self.number_of_states == 5:
            if 'dressing2' not in self.config["states"].keys(): # if not dressing with two states
                # ground state mapping
                # bit of a hack - find a better way to do this
                self.lifetimes[2] = self.lifetimes[0]
        
        self.decay_rates = 1/self.lifetimes
        
        if self.Lq == 0: # storage state is in ground state manifold
            self.decay_rates[1] = 0 # set spin wave decay to be zero
        elif self.number_of_states == 5:
            if 'dressing2' not in self.config["states"].keys(): # if not dressing with two states
                self.decay_rates[3] = 0 # set spin wave decay of ground state to be zero

        self.gammas = self.decay_rates/2   

    def Hhfs(self, J, I):
        """Provides the I dot J matrix (hyperfine structure interaction)"""
        gJ=int(2*J+1)
        Jx=self.jx(J)
        Jy=self.jy(J)
        Jz=self.jz(J)
        Ji=np.identity(gJ)
        J2=np.dot(Jx,Jx)+np.dot(Jy,Jy)+np.dot(Jz,Jz)

        gI=int(2*I+1)
        gF=gJ*gI
        Ix=self.jx(I)
        Iy=self.jy(I)
        Iz=self.jz(I)
        Ii=np.identity(gI)
        Fx=kron(Jx,Ii)+kron(Ji,Ix)
        Fy=kron(Jy,Ii)+kron(Ji,Iy)
        Fz=kron(Jz,Ii)+kron(Ji,Iz)
        Fi=np.identity(gF)
        F2=np.dot(Fx,Fx)+np.dot(Fy,Fy)+np.dot(Fz,Fz)
        Hhfs=0.5*(F2-I*(I+1)*Fi-kron(J2,Ii))
        return Hhfs

    def Bbhfs(self, J, I):
        """Calculates electric quadrupole matrix.
        """
        gJ=int(2*J+1)
        Jx=self.jx(J)
        Jy=self.jy(J)
        Jz=self.jz(J)

        gI=int(2*I+1)
        gF=gJ*gI
        Ix=self.jx(I)
        Iy=self.jy(I)
        Iz=self.jz(I)
        
        Fi=np.identity(gF)

        IdotJ=kron(Jx,Ix)+kron(Jy,Iy)+kron(Jz,Iz)
        IdotJ2=np.dot(IdotJ,IdotJ)

        if I != 0:
            Bbhfs=1./(6*I*(2*I-1))*(3*IdotJ2+3./2*IdotJ-I*(I+1)*15./4*Fi)
        else:
            Bbhfs = 0
        return Bbhfs

    def _jp(self, jj):
        b = 0
        dim = int(2*jj+1)
        jp = np.zeros((dim,dim))
        z = np.arange(dim)
        m = jj-z
        while b<dim-1:
            mm = m[b+1]
            jp[b,b+1] = np.sqrt(jj*(jj+1)-mm*(mm+1)) 
            b = b+1
        return jp

    def jx(self, jj):
        jp = self._jp(jj)
        jm=np.transpose(jp)
        jx=0.5*(jp+jm)
        return jx

    def jy(self, jj):
        jp = self._jp(jj)
        jm=np.transpose(jp)
        jy=0.5j*(jm-jp)
        return jy

    def jz(self, jj):
        jp = self._jp(jj)
        jm=np.transpose(jp)
        jz=0.5*(np.dot(jp,jm)-np.dot(jm,jp))
        return jz

    def hyperfine_splitting(self):
        # calculate from hyperfine constant?
        # or just lookup?

        # ground state hyperfine splitting
        df = pd.read_csv(self.filename_hyperfine)

        Ag = df[df['State'].str.contains(self.ground_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
        Bg = df[df['State'].str.contains(self.ground_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

        H = Ag*self.Hhfs(self.Jg, self.I)
        values = eigh(H)[0].real
        self.deltaHF = abs(values[-1] - values[0])
        if self.config["states"]["initial"]["F"] == self.hyperfine_states(self.I, self.Jg)[1]:
            self.deltaHF *= -1
            
        if self.splitting == False:
            self.wg = np.array([0])
            self.wj = np.array([0])
            self.wq = np.array([0])
            if self.number_of_states == 4:
                self.wb = np.array([0])
            if self.number_of_states == 5:
                if 'dressing2' in self.config["states"].keys(): # dressing with two states
                    self.wb = np.array([0])
                    self.wb2 = np.array([0])
                    self.dress_state_splitting = (self.angular_frequencies[2] - self.angular_frequencies[3])/(2*np.pi)
                else:
                    # ground state mapping
                    self.wb = np.array([0])

        else:
            df = pd.read_csv(self.filename_hyperfine)

            Ag = df[df['State'].str.contains(self.ground_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
            Bg = df[df['State'].str.contains(self.ground_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

            Aj = df[df['State'].str.contains(self.intermediate_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
            Bj = df[df['State'].str.contains(self.intermediate_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

            Aq = df[df['State'].str.contains(self.storage_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
            Bq = df[df['State'].str.contains(self.storage_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

            if len(self.Fg)>1:
                H = Ag*self.Hhfs(self.Jg, self.I)
                values = eigh(H)[0].real
                indices = np.concatenate(([0], np.cumsum(2*self.Fg[:-1]+1))).astype(int)
                if Ag < 0:
                    self.wg = np.flip(values)[indices]
                else:
                    self.wg = values[indices]
            else:
                self.wg = np.array([0])


            if len(self.Fj)>1:
                dim = int((2*self.Lj+1)*(2*self.S+1)*(2*self.I+1))  # total dimension of matrix
                H = Aj * self.Hhfs(self.Jj, self.I) + Bj*self.Bbhfs(self.Jj,self.I)
                values = eigh(H)[0].real
                indices = np.concatenate(([0], np.cumsum(2*self.Fj[:-1]+1))).astype(int)
                if Aj < 0:
                    self.wj = np.flip(values)[indices]
                else:
                    self.wj = values[indices]
                
            else:
                self.wj = np.array([0])

            if len(self.Fq)>1:
                H = Aq * self.Hhfs(self.Jq, self.I) + Bq*self.Bbhfs(self.Jq,self.I)
                values = eigh(H)[0].real
                indices = np.concatenate(([0], np.cumsum(2*self.Fq[:-1]+1))).astype(int)
                if Aq < 0:
                    self.wq = np.flip(values)[indices]
                else:
                    self.wq = values[indices]
            else:
                self.wq = np.array([0])

            if self.number_of_states == 4:
                Ab = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
                Bb = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
                if len(self.Fb)>1:
                    H = Ab * self.Hhfs(self.Jb, self.I) + Bb*self.Bbhfs(self.Jb,self.I)
                    values = eigh(H)[0].real
                    indices = np.concatenate(([0], np.cumsum(2*self.Fb[:-1]+1))).astype(int)
                    if Ab < 0:
                        self.wb = np.flip(values)[indices]
                    else:
                        self.wb = values[indices]
                else:
                    self.wb = np.array([0])

            if self.number_of_states == 5:
                if 'dressing2' in self.config["states"].keys(): # dressing with two states
                    Ab = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
                    Bb = df[df['State'].str.contains(self.dressing_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
                    Ab2 = df[df['State'].str.contains(self.dressing_state2)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
                    Bb2 = df[df['State'].str.contains(self.dressing_state2)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
                    if len(self.Fb)>1:
                        H = Ab * self.Hhfs(self.Jb, self.I) + Bb*self.Bbhfs(self.Jb,self.I)
                        values = eigh(H)[0].real
                        indices = np.concatenate(([0], np.cumsum(2*self.Fb[:-1]+1))).astype(int)
                        if Ab < 0:
                            self.wb = np.flip(values)[indices]
                        else:
                            self.wb = values[indices]
                    else:
                        self.wb = np.array([0])

                    if len(self.Fb2)>1:
                        H = Ab2 * self.Hhfs(self.Jb2, self.I) + Bb2*self.Bbhfs(self.Jb2,self.I)
                        values = eigh(H)[0].real
                        indices = np.concatenate(([0], np.cumsum(2*self.Fb2[:-1]+1))).astype(int)
                        if Ab2 < 0:
                            self.wb2 = np.flip(values)[indices]
                        else:
                            self.wb2 = values[indices]
                    else:
                        self.wb2 = np.array([0])
                        
                    self.dress_state_splitting = (self.angular_frequencies[2] - self.angular_frequencies[3])/(2*np.pi)
                else:
                    # ground state mapping
                    Ab = Ag
                    Bb = Bg
                    if len(self.Fb)>1:
                        H = Ab * self.Hhfs(self.Jb, self.I) + Bb*self.Bbhfs(self.Jb,self.I)
                        values = eigh(H)[0].real
                        indices = np.concatenate(([0], np.cumsum(2*self.Fb[:-1]+1))).astype(int)
                        if Ab < 0:
                            self.wb = np.flip(values)[indices]
                        else:
                            self.wb = values[indices]
                    else:
                        self.wb = np.array([0])
            
    
    def Wigner6jPrefactorSteck(self, Fdash, J, I):
        return pow(-1, -Fdash+J+1+I) * np.sqrt(( 2*Fdash + 1 ) * (2*J + 1) )

    def transition_strength(self, I, J, F, mF, Jdash, Fdash, mFdash, q): #in terms of reduced dipole moment (J)
        element = ( float(CG(Fdash, mFdash, 1, -q, F, mF).doit())
                *  float( Wigner6j(J, Jdash, 1, Fdash, F, I).doit() ) * self.Wigner6jPrefactorSteck(Fdash, J, I) )
        return element
    
    def coupling(self, J, F, mF, Jdash, Fdash, mFdash):
        """Makes couplnig tensor"""
        Q = np.array([-1, 1])
        coupling_tensor = np.zeros((len(F), len(mF), len(Fdash), len(mFdash), len(Q)))
        if self.splitting == False:
            coupling_tensor[:, :, :, :, 0] = 1
        else:
            #transition_strength function uses sympy which I have't been able to vectorise
            for Fi, F_ in enumerate(F):
                for mFi, mF_ in enumerate(mF):
                    for Fdashi, Fdash_ in enumerate(Fdash):
                        for mFdashi, mFdash_ in enumerate(mFdash):
                            for Qi, Q_ in enumerate(Q):
                                coupling_tensor[Fi, mFi, Fdashi, mFdashi, Qi] = self.transition_strength(self.I, J, F_, mF_, Jdash, Fdash_, mFdash_, Q_)

        return coupling_tensor
    
    def rabi_frequency_to_power(self, Omega, r, index):
        # index := which dipole strength to use 
        # returns power in W
        P = c*epsilon_0*pow(hbar,2)*np.pi/2 * pow(r,2) * pow(Omega/self.reduced_dipoles[index], 2)
        return P
    
    def rabi_frequency_to_intensity(self, Omega, index):
        # index := which dipole strength to use 
        I = c*epsilon_0*pow(hbar,2)/2 * pow(Omega/self.reduced_dipoles[index], 2)
        return I
    
    def control_pulse_to_energy(self, Control, t, r, index):
        """ Takes list representing Control(t), returns energy contained within pulse in Joules"""
        # Control could have two polarisations
        # Control must be in rabi frequency (Hz)
        # t must be in seconds
        E = np.trapz(c*epsilon_0*pow(hbar,2)/2 * pow(Control/self.reduced_dipoles[index], 2), x=t, axis=0) * np.pi * pow(r, 2)
        return E
    

class Rb87(Alkalis):
    def __init__(self, config):

        self.atom = 'Rb87'
        
        self.I  = 1.5 
        self.mass = 86.909180520*amu
        self.ng = 5

        self.config = config

        self.filename = '/home/otps3141/Documents/Dokumente/ETH QE/Master Thesis Imperial/Thesis/Code/OBEsimulation/Rb/Rb1MatrixElements.csv'
        self.filename_hyperfine =  '/home/otps3141/Documents/Dokumente/ETH QE/Master Thesis Imperial/Thesis/Code/OBEsimulation/Rb/Rb87_hyperfine.csv'

        super().__init__()

class Rb85(Alkalis):
    def __init__(self, config):

        self.atom = 'Rb85'
        
        self.I  = 2.5 
        self.mass = 84.9117897*amu
        self.ng = 5

        self.config = config

        self.filename = '/home/otps3141/Documents/Dokumente/ETH QE/Master Thesis Imperial/Thesis/Code/OBEsimulation/Rb/Rb1MatrixElements.csv'
        self.filename_hyperfine =  '/home/otps3141/Documents/Dokumente/ETH QE/Master Thesis Imperial/Thesis/Code/OBEsimulation/Rb/Rb85_hyperfine.csv'

        super().__init__()

class Cs(Alkalis):
    def __init__(self, config):

        self.atom = 'Cs'
        
        self.I  = 3.5 
        self.mass = 132.905451931*amu
        self.ng = 6

        self.config = config

        self.filename = 'Cs\\Cs1MatrixElements.csv'
        self.filename_hyperfine =  'Cs\\Cs_hyperfine.csv'

        super().__init__()

# # K

# # Na

# # Li





# import numpy as np
# import pandas as pd
# from scipy.constants import physical_constants, epsilon_0, hbar, c, e, h
# from scipy.linalg import kron, eigh
# from sympy.physics.quantum.cg import Wigner6j
# from sympy.physics.quantum.cg import CG
# from fractions import Fraction
# amu = physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
# au = physical_constants['atomic unit of electric dipole mom.'][0] # atomic units for reduced dipole moment

# class Alkalis:
#     def __init__(self):
#         self.S = 1/2
#         self.Lg = 0
#         self.Jg = 1/2
#         self.Lj = 1
        

#     def hyperfine_states(self, I, J):
#         Fmin = np.abs(J - I)
#         Fmax = np.abs(J + I)
#         F = np.arange(Fmin, Fmax+1)
#         return F

#     def momentum_letter(self, L):
#         if L == 0:
#             return 's'
#         elif L == 1:
#             return 'p'
#         elif L == 2:
#             return 'd'
#         elif L == 3:
#             return 'f'
        
#     def order_states(self, nlist, Llist, Jlist):
#         order = np.lexsort((nlist, Llist))
#         ordered_ground_states = Jlist[order][:-1]
#         ordered_excited_states = Jlist[order][1:]
#         return ordered_ground_states, ordered_excited_states
        
#     def unpack_config(self, I, ng, config):
        
#         self.number_of_states = 3

#         if config["Hyperfine splitting?"] == False:
#             self.Fg = np.array([0])
#             self.mg = np.array([0])

#             self.Fj = np.array([0])
#             self.mj = np.array([0])
#             self.nj = config["states"]["intermediate"]["n"]
#             self.Jj = config["states"]["intermediate"]["J"]

#             self.Fq = np.array([0])
#             self.mq = np.array([0])
#             if config["states"]["storage"]["L"] == 0: # using ground state to store i.e. lambda scheme
#                 self.nq = self.ng
#                 self.Jq = self.Jg
#                 self.Lq = self.Lg
#             else:
#                 self.nq = config["states"]["storage"]["n"]
#                 self.Jq = config["states"]["storage"]["J"]
#                 self.Lq = config["states"]["storage"]["L"]

#             # dressing state?
#             if len(config["states"]) == 4:
#                 self.number_of_states += 1
#                 self.nb = config["states"]["dressing"]["n"]
#                 self.Lb = config["states"]["dressing"]["L"]
#                 self.Jb = config["states"]["dressing"]["J"]
#                 self.Fb = np.array([0])
#                 self.mb = np.array([0])
            


#         else:
#             # Ground state
#             self.Fg = config["states"]["initial"]["F"]
#             if isinstance(self.Fg, int):
#                 self.Fg = np.array([self.Fg])
#             self.mg = np.arange(-max(self.Fg), max(self.Fg)+1)

#             # intermediate state
#             self.nj = config["states"]["intermediate"]["n"]
#             self.Jj = config["states"]["intermediate"]["J"]
#             self.Fj = self.hyperfine_states(I, self.Jj)
#             self.mj = np.arange(-max(self.Fj), max(self.Fj)+1)

#             # storage state
#             if config["states"]["storage"]["L"] == 0: # using ground state to store i.e. lambda scheme
#                 # could have same F as initial state, need to be pumped into initial mF state!
#                 self.nq = ng
#                 self.Lq = self.Lg
#                 self.Jq = self.Jg
#                 self.Fq = config["states"]["storage"]["F"]
#                 if isinstance(self.Fq, int):
#                     self.Fq = np.array([self.Fq])
#                 self.mq = np.arange(-max(self.Fq), max(self.Fq)+1)
#             else:
#                 self.nq = config["states"]["storage"]["n"]
#                 self.Lq = config["states"]["storage"]["L"]
#                 self.Jq = config["states"]["storage"]["J"]
#                 self.Fq = self.hyperfine_states(I, self.Jq)
#                 self.mq = np.arange(-max(self.Fq), max(self.Fq)+1)

#             # dressing state?
#             if len(config["states"]) == 4:
#                 self.number_of_states += 1
#                 self.nb = config["states"]["dressing"]["n"]
#                 self.Lb = config["states"]["dressing"]["L"]
#                 self.Jb = config["states"]["dressing"]["J"]
#                 self.Fb = self.hyperfine_states(I, self.Jb)
#                 self.mb = np.arange(-max(self.Fb), max(self.Fb)+1)

#             # ground state mapping states?
#             # self.number_of_states += 1

#     def lookup_atom_constants(self, filename):
#         ground_state = str(self.ng)+self.momentum_letter(self.Lg)+str(Fraction(self.Jg))
#         intermediate_state = str(self.nj)+self.momentum_letter(self.Lj)+str(Fraction(self.Jj))
#         storage_state = str(self.nq)+self.momentum_letter(self.Lq)+str(Fraction(self.Jq))

#         # ordered_ground_states, ordered_excited_states = self.order_states(np.array([self.ng, self.nj, self.nq]), 
#         #                                                                   np.array([self.Lg, self.Lj, self.Lq]),
#         #                                                                   np.array([self.Jg, self.Jj, self.Jq]))

#         df = pd.read_csv(filename)

#         wavelengths = np.zeros(self.number_of_states-1)
#         reduced_dipoles = np.zeros(self.number_of_states-1)

#         self.J_array = np.zeros(self.number_of_states)
#         self.J_array[0] = self.Jg
#         self.J_array[1] = self.Jj
#         self.J_array[2] = self.Jq
        
#         self.reduced_dipoles = np.zeros(self.number_of_states - 1)
#         self.lifetimes = np.zeros(self.number_of_states - 1)

#         # make into loop
#         query = df[
#                         (df['Initial'].str.contains(ground_state) & df['Final'].str.contains(intermediate_state)) |
#                         (df['Final'].str.contains(ground_state) & df['Initial'].str.contains(intermediate_state))
#                     ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

#         [[wavelengths[0], reduced_dipoles[0]]] = query

#         query = df[
#                         (df['Initial'].str.contains(intermediate_state) & df['Final'].str.contains(storage_state)) |
#                         (df['Final'].str.contains(intermediate_state) & df['Initial'].str.contains(storage_state))
#                     ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

#         [[wavelengths[1], reduced_dipoles[1]]] = query

#         self.reduced_dipoles[0] = reduced_dipoles[0]*au/np.sqrt(2*self.Jg+1) # to make same convention as Steck
#         self.reduced_dipoles[1] = reduced_dipoles[1]*au/np.sqrt(2*self.Jj+1)

#         if self.number_of_states == 4:
#             # dressing
#             dressing_state = str(self.nb)+self.momentum_letter(self.Lb)+str(Fraction(self.Jb))
#             query = df[
#                         (df['Initial'].str.contains(storage_state) & df['Final'].str.contains(dressing_state)) |
#                         (df['Final'].str.contains(storage_state) & df['Initial'].str.contains(dressing_state))
#                     ][['Wavelength (nm)', 'Matrix element (a.u.)']].to_numpy()

#             [[wavelengths[2], reduced_dipoles[2]]] = query
#             self.reduced_dipoles[2] = reduced_dipoles[2]*au/np.sqrt(2*self.Jq+1)
#             self.J_array[3] = self.Jb


#         elif self.number_of_states == 5:
#             # ground state mapping
#             pass

#         self.wavelengths = wavelengths*1e-9 # convert to m
#         self.angular_frequencies = 2*np.pi*c/self.wavelengths
#         self.wavevectors = 2*np.pi/self.wavelengths

#         self.lifetimes = ( 3*np.pi*epsilon_0*hbar*pow(c,3) * 
#              (2*self.J_array[1:] + 1)/(pow(self.angular_frequencies,3)*(2*self.J_array[:-1] + 1)*pow(self.reduced_dipoles,2)) )
        
#         # self.lifetimes[0] = ( 3*np.pi*epsilon_0*hbar*pow(c,3) * 
#         #      (2*self.Jj + 1)/(pow(self.angular_frequencies[0],3)*(2*self.Jg + 1)*pow(self.reduced_dipoles[0],2))
#         #             )
#         # self.lifetimes[1] = ( 3*np.pi*epsilon_0*hbar*pow(c,3) * 
#         #      (2*self.Jq + 1)/(pow(self.angular_frequencies[1],3)*(2*self.Jj + 1)*pow(self.reduced_dipoles[1],2))
#         #             )
        
#         self.decay_rates = 1/self.lifetimes
        
#         if self.Lq == 0: # storage state is in ground state manifold
#             self.decay_rates[1] = 0 # set spin wave decay to be zero

#         self.gammas = self.decay_rates/2       


#     def hyperfine_splitting(self, hyperfine_splitting, filename):
#         # calculate from hyperfine constant?
#         # or just lookup?

#         ground_state = str(self.ng)+self.momentum_letter(self.Lg)+str(Fraction(self.Jg))
#         intermediate_state = str(self.nj)+self.momentum_letter(self.Lj)+str(Fraction(self.Jj))
#         storage_state = str(self.nq)+self.momentum_letter(self.Lq)+str(Fraction(self.Jq))

#         if hyperfine_splitting == False:
#             self.wg = np.array([0])
#             self.wj = np.array([0])
#             self.wq = np.array([0])
#             if self.number_of_states == 4:
#                 self.wb = np.array([0])

#         else:
#             df = pd.read_csv(filename)
#             Ag = df[df['State'].str.contains(ground_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
#             Aj = df[df['State'].str.contains(intermediate_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
#             Aq = df[df['State'].str.contains(storage_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
#             Bg = df[df['State'].str.contains(ground_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
#             Bj = df[df['State'].str.contains(intermediate_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
#             Bq = df[df['State'].str.contains(storage_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6

#             if len(self.Fg)>1:
#                 H = Ag*self.Hhfs(self.Jg, self.I)
#                 values = eigh(H)[0].real
#                 indices = np.concatenate(([0], np.cumsum(2*self.Fg[:-1]+1))).astype(int)
#                 if Ag < 0:
#                     self.wg = np.flip(values)[indices]
#                 else:
#                     self.wg = values[indices]
#             else:
#                 self.wg = np.array([0])


#             if len(self.Fj)>1:
#                 dim = int((2*self.Lj+1)*(2*self.S+1)*(2*self.I+1))  # total dimension of matrix
#                 H = Aj * self.Hhfs(self.Jj, self.I) + Bj*self.Bbhfs(self.Jj,self.I)
#                 values = eigh(H)[0].real
#                 indices = np.concatenate(([0], np.cumsum(2*self.Fj[:-1]+1))).astype(int)
#                 if Aj < 0:
#                     self.wj = np.flip(values)[indices]
#                 else:
#                     self.wj = values[indices]
                
#             else:
#                 self.wj = np.array([0])

#             if len(self.Fq)>1:
#                 H = Aq * self.Hhfs(self.Jq, self.I) + Bq*self.Bbhfs(self.Jq,self.I)
#                 values = eigh(H)[0].real
#                 indices = np.concatenate(([0], np.cumsum(2*self.Fq[:-1]+1))).astype(int)
#                 if Aq < 0:
#                     self.wq = np.flip(values)[indices]
#                 else:
#                     self.wq = values[indices]
#             else:
#                 self.wq = np.array([0])

#             if self.number_of_states == 4:
#                 dressing_state = str(self.nb)+self.momentum_letter(self.Lb)+str(Fraction(self.Jb))
#                 Ab = df[df['State'].str.contains(dressing_state)]['Hyperfine constant (A)'].to_numpy()[0]*1e6
#                 Bb = df[df['State'].str.contains(dressing_state)]['Hyperfine constant (B)'].to_numpy()[0]*1e6
#                 if len(self.Fb)>1:
#                     H = Ab * self.Hhfs(self.Jb, self.I) + Bb*self.Bbhfs(self.Jb,self.I)
#                     values = eigh(H)[0].real
#                     indices = np.concatenate(([0], np.cumsum(2*self.Fb[:-1]+1))).astype(int)
#                     if Ab < 0:
#                         self.wb = np.flip(values)[indices]
#                     else:
#                         self.wb = values[indices]
#                 else:
#                     self.wb = np.array([0])

#         # ground_state = str(self.ng)+self.momentum_letter(self.Lg)+str(self.Jg)
#         # intermediate_state = str(self.nj)+self.momentum_letter(self.Lj)+str(self.Jj)
#         # storage_state = str(self.nq)+self.momentum_letter(self.Lq)+str(self.Jq)

#         # df = pd.read_csv(filename)

#         # if len(self.Fg)>1:
#         #     self.wg = 2*np.pi*1e6*np.array(df[df['State'].str.contains(ground_state)]['Hyperfine states'].to_numpy()[0][1:-1].split(','))
#         # else:
#         #     self.wg = np.array([0])


#         # if len(self.Fj)>1:
#         #     self.wj = 2*np.pi*1e6*np.array(df[df['State'].str.contains(intermediate_state)]['Hyperfine states'].to_numpy()[0][1:-1].split(','))
#         # else:
#         #     self.wj = np.array([0])

#         # if len(self.Fq)>1:
#         #     self.wq = 2*np.pi*1e6*np.array(df[df['State'].str.contains(storage_state)]['Hyperfine states'].to_numpy()[0][1:-1].split(','))
#         # else:
#         #     self.wq = np.array([0])

#         # dressing
#         # ground state mapping
            
#     def Hhfs(self, J, I):
#         """Provides the I dot J matrix (hyperfine structure interaction)"""
#         gJ=int(2*J+1)
#         Jx=self.jx(J)
#         Jy=self.jy(J)
#         Jz=self.jz(J)
#         Ji=np.identity(gJ)
#         J2=np.dot(Jx,Jx)+np.dot(Jy,Jy)+np.dot(Jz,Jz)

#         gI=int(2*I+1)
#         gF=gJ*gI
#         Ix=self.jx(I)
#         Iy=self.jy(I)
#         Iz=self.jz(I)
#         Ii=np.identity(gI)
#         Fx=kron(Jx,Ii)+kron(Ji,Ix)
#         Fy=kron(Jy,Ii)+kron(Ji,Iy)
#         Fz=kron(Jz,Ii)+kron(Ji,Iz)
#         Fi=np.identity(gF)
#         F2=np.dot(Fx,Fx)+np.dot(Fy,Fy)+np.dot(Fz,Fz)
#         Hhfs=0.5*(F2-I*(I+1)*Fi-kron(J2,Ii))
#         return Hhfs

#     def Bbhfs(self, J, I):
#         """Calculates electric quadrupole matrix.
#         """
#         gJ=int(2*J+1)
#         Jx=self.jx(J)
#         Jy=self.jy(J)
#         Jz=self.jz(J)

#         gI=int(2*I+1)
#         gF=gJ*gI
#         Ix=self.jx(I)
#         Iy=self.jy(I)
#         Iz=self.jz(I)
        
#         Fi=np.identity(gF)

#         IdotJ=kron(Jx,Ix)+kron(Jy,Iy)+kron(Jz,Iz)
#         IdotJ2=np.dot(IdotJ,IdotJ)

#         if I != 0:
#             Bbhfs=1./(6*I*(2*I-1))*(3*IdotJ2+3./2*IdotJ-I*(I+1)*15./4*Fi)
#         else:
#             Bbhfs = 0
#         return Bbhfs

#     def _jp(self, jj):
#         b = 0
#         dim = int(2*jj+1)
#         jp = np.zeros((dim,dim))
#         z = np.arange(dim)
#         m = jj-z
#         while b<dim-1:
#             mm = m[b+1]
#             jp[b,b+1] = np.sqrt(jj*(jj+1)-mm*(mm+1)) 
#             b = b+1
#         return jp

#     def jx(self, jj):
#         jp = self._jp(jj)
#         jm=np.transpose(jp)
#         jx=0.5*(jp+jm)
#         return jx

#     def jy(self, jj):
#         jp = self._jp(jj)
#         jm=np.transpose(jp)
#         jy=0.5j*(jm-jp)
#         return jy

#     def jz(self, jj):
#         jp = self._jp(jj)
#         jm=np.transpose(jp)
#         jz=0.5*(np.dot(jp,jm)-np.dot(jm,jp))
#         return jz

#     def Wigner6jPrefactorSteck(self, Fdash, J, I):
#         return pow(-1, -Fdash+J+1+I) * np.sqrt(( 2*Fdash + 1 ) * (2*J + 1) )

#     def transition_strength(self, I, J, F, mF, Jdash, Fdash, mFdash, q): #in terms of reduced dipole moment (J)
#         element = ( float(CG(Fdash, mFdash, 1, -q, F, mF).doit())
#                 *  float( Wigner6j(J, Jdash, 1, Fdash, F, I).doit() ) * self.Wigner6jPrefactorSteck(Fdash, J, I) )
#         return element
    
#     def coupling(self, hyperfine_splitting, J, F, mF, Jdash, Fdash, mFdash):
#         """Makes couplnig tensor"""
#         Q = np.array([-1, 1])
#         coupling_tensor = np.zeros((len(F), len(mF), len(Fdash), len(mFdash), len(Q)))
#         if hyperfine_splitting == False:
#             coupling_tensor[:, :, :, :, 0] = 1
#         else:
#             #transition_strength function uses sympy which I have't been able to vectorise
#             for Fi, F_ in enumerate(F):
#                 for mFi, mF_ in enumerate(mF):
#                     for Fdashi, Fdash_ in enumerate(Fdash):
#                         for mFdashi, mFdash_ in enumerate(mFdash):
#                             for Qi, Q_ in enumerate(Q):
#                                 coupling_tensor[Fi, mFi, Fdashi, mFdashi, Qi] = self.transition_strength(self.I, J, F_, mF_, Jdash, Fdash_, mFdash_, Q_)

#         return coupling_tensor
    

# class Rb87(Alkalis):
#     def __init__(self, config):
#         super().__init__()
#         self.I  = 1.5 
#         self.mass = 86.909180520*amu
#         self.FS = 7.123 # Fine-structure splitting (MHz)
#         isotope_shift = -56 # MHz

#         self.ng = 5
#         self.filename = 'Rb\\Rb1MatrixElements.csv'
#         self.filename_hyperfine =  'Rb\\Rb87_hyperfine.csv'

#         self.unpack_config(self.I, self.ng, config)

#         self.lookup_atom_constants(self.filename)

#         # hyperfine splitting - calculate?
#         self.hyperfine_splitting(config["Hyperfine splitting?"], self.filename_hyperfine)

#         # coupling tensor - separate ones for ge, es, etc?
#         self.coupling_ge = self.coupling(config["Hyperfine splitting?"], self.Jg, self.Fg, self.mg, self.Jj, self.Fj, self.mj)
#         self.coupling_es = self.coupling(config["Hyperfine splitting?"], self.Jj, self.Fj, self.mj, self.Jq, self.Fq, self.mq)
#         # dressing
#         if self.number_of_states == 4:
#             self.coupling_sb = self.coupling(config["Hyperfine splitting?"], self.Jq, self.Fq, self.mq, self.Jb, self.Fb, self.mb)
#         # ground state mapping
        

# class Rb85(Alkalis):
#     def __init__(self, config):
#         super().__init__()
#         self.I  = 2.5 
#         self.mass = 84.9117897*amu

#         self.ng = 5
#         self.filename = 'Rb\\Rb1MatrixElements.csv'
#         self.filename_hyperfine =  'Rb\\Rb85_hyperfine.csv'

#         self.unpack_config(self.I, self.ng, config)

#         self.lookup_atom_constants(self.filename)

#         # hyperfine splitting - calculate?
#         self.hyperfine_splitting(self.filename_hyperfine)

#         # coupling tensor - separate ones for ge, es, etc?
#         self.coupling_ge = self.coupling(config["Hyperfine splitting?"], self.Jg, self.Fg, self.mg, self.Jj, self.Fj, self.mj)
#         self.coupling_es = self.coupling(config["Hyperfine splitting?"], self.Jj, self.Fj, self.mj, self.Jq, self.Fq, self.mq)
#         # dressing
#         if self.number_of_states == 4:
#             self.coupling_sb = self.coupling(config["Hyperfine splitting?"], self.Jq, self.Fq, self.mq, self.Jb, self.Fb, self.mb)
#         # ground state mapping
        

# class Cs(Alkalis):
#     def __init__(self, config):
#         super().__init__()
#         self.I  = 3.5
#         self.mass = 132.905451931*amu

#         self.ng = 6
#         self.filename = 'Cs\\Cs1MatrixElements.csv'
#         self.filename_hyperfine =  'Cs\\Cs_hyperfine.csv'

#         self.unpack_config(self.I, self.ng, config)

#         self.lookup_atom_constants(self.filename)

#         # hyperfine splitting - calculate?
#         self.hyperfine_splitting(self.filename_hyperfine)

#         # coupling tensor - separate ones for ge, es, etc?
#         self.coupling_ge = self.coupling(config["Hyperfine splitting?"], self.Jg, self.Fg, self.mg, self.Jj, self.Fj, self.mj)
#         self.coupling_es = self.coupling(config["Hyperfine splitting?"], self.Jj, self.Fj, self.mj, self.Jq, self.Fq, self.mq)
#         # dressing
#         if self.number_of_states == 4:
#             self.coupling_sb = self.coupling(config["Hyperfine splitting?"], self.Jq, self.Fq, self.mq, self.Jb, self.Fb, self.mb)
#         # ground state mapping

# # K

# # Na

# # Li

