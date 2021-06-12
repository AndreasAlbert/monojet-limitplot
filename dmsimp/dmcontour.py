import pickle
import numpy as np
from mediator_width import *
from scipy.interpolate import interp1d
import itertools
from skimage.measure import find_contours
from limitlib import interpolate_rbf

def fdm_analytic(mmed, mdm, coupling):
    '''Correction factor BR(mmed, mdm) / BR(mmed, mdm=1 GeV)'''
    ret = []
    for mmed, mdm in zip(mmed, mdm):
        if coupling=='vector':
            reference = gamma_vector_chi(mmed, m_x=0, g_chi=1.0) / gamma_vector_total(mmed, m_x=0, g_chi=1.0, g_q=0.25)
            new = gamma_vector_chi(mmed, m_x=mdm, g_chi=1.0) / gamma_vector_total(mmed, m_x=mdm, g_chi=1.0, g_q=0.25)
        elif coupling == 'axial':
            reference = gamma_axial_chi(mmed, m_x=0, g_chi=1.0) / gamma_axial_total(mmed, m_x=0, g_chi=1.0, g_q=0.25)
            new = gamma_axial_chi(mmed, m_x=mdm, g_chi=1.0) / gamma_axial_total(mmed, m_x=mdm, g_chi=1.0, g_q=0.25)
        elif coupling == 'pseudoscalar':
            reference = gamma_pseudo_chi(mmed, m_x=0, g_chi=1.0) / gamma_pseudo_total(mmed, m_x=0, g_chi=1.0, g_q=0.25)
            new = gamma_pseudo_chi(mmed, m_x=mdm, g_chi=1.0) / gamma_pseudo_total(mmed, m_x=mdm, g_chi=1.0, g_q=0.25)
        if new > 0:
            ret.append(reference/new)
        else:
            ret.append(np.nan)
    return ret
class OffshellGenerator():
    def __init__(self, coupling):
        with open(f"dmsimp/input/offshell_functions/offshell_functions_{coupling}_2016.pkl","rb") as f:
            self.functions = pickle.load(f)
        self.coupling = coupling
        self.filter_functions()
        self.mode = 'linearinterp'
    def filter_functions(self):
        if self.coupling == 'vector':
            good = lambda x: x > 200
        else:
            good = lambda x: x > 150

        self.functions = {
            k : v for k,v in self.functions.items() if good(k)
        }

    def evaluate_correction(self,mmed, mdm):
        if self.mode=='nearest':
            distance = 1e9
            best_mmed = -1
            for immed in self.functions.keys():
                idistance = np.abs(immed - mmed)
                if idistance < distance:
                    distance = idistance
                    best_mmed = immed

            corr = self.functions[best_mmed](mdm/mmed)
        if self.mode=='linearinterp':
            keys = sorted(list(self.functions.keys()))

            nkeys = len(keys)
            if mmed < keys[0]:
                m1 = keys[0]
                m2 = keys[0]
            else:
                for i in range(nkeys):
                    if i==nkeys-1:
                        m1 = keys[i]
                        m2 = m1
                    else:
                        if mmed > keys[i] and mmed < keys[i+1]:
                            m1 = keys[i]
                            m2 = keys[i+1]
                            break

            if m2 == m1:
                fac1 = 1
                fac2 = 0
            else:
                fac1 = (m2-mmed) / (m2-m1)
                fac2 = 1 - fac1

            val1 = self.functions[m1](mdm/mmed)
            val2 = self.functions[m2](mdm/mmed)
            # if val1 > 1e5:
            #     fac1 = 0
            #     fac2 = 1
            # elif val2 > 1e5:
            #     fac1 = 1
            #     fac2 = 0

            corr = fac1 * val1 + fac2 * val2
        return corr

class DMInterp():
    def __init__(self, df, quantile='exp'):
        self.df = df
        self.quantile=quantile

        couplings = set(df.coupling)
        assert(len(couplings)==1)
        self.coupling = couplings.pop()

        if self.coupling in ['axial', 'vector']:
            self.offshell = OffshellGenerator(self.coupling)
        else:
            self.offshell = None

        x = df[df.mdm==1].mmed
        y = np.log(df[df.mdm==1][quantile])
        f = interp1d(
                    x,
                    y,
                    fill_value='extrapolate'
                    )
        self.mu_1d_mmed = lambda x: np.exp(f(x))
        self.calculate_grid()

    def mdm_correction_factor(self, mmed, mdm):
        '''Correction factor for going from mdm~0 to finite mdm'''
        if self.coupling in ['axial','vector']:
            mdm_threshold = 0.45
        else:
            mdm_threshold = 0.499

        factor = fdm_analytic(
                            mmed=itertools.repeat(mmed),
                            mdm=[min(mdm, mdm_threshold*mmed)],
                            coupling=self.coupling
                            )[0]


        if mdm/mmed > mdm_threshold:
            if self.offshell:
                factor *= self.offshell.evaluate_correction(mmed, mdm)
            else:
                factor *= np.exp(50*((mdm/mmed) / mdm_threshold-1))
        return factor

    def mu_2d(self, mmed, mdm):
        mu = self.mu_1d_mmed(mmed)
        mu *= self.mdm_correction_factor(mmed, mdm)
        return mu

    def mdm_fractions_for_grid(self,mmed):
        if self.coupling in ['axial','vector']:
            mdmfracmax = max(5 - 4 * (mmed-100) / 400, 0.75)
            fractions =  list(np.linspace(0.,0.35,8))
            fractions += list(np.linspace(0.35,0.6,10))
            fractions += list(
                            mdmfracmax - np.linspace(
                                                    mdmfracmax - 0.6,
                                                    0,
                                                    40)
                            )
        else:
            fractions = [0, 1./mmed, 0.01,0.05,0.1,0.15,0.2,0.25,0.3] + list(np.linspace(0.3,0.6,30)) + list(np.linspace(0.6,1,5))
        return sorted(fractions)
    def calculate_grid(self):
        x, y, z = [], [], []

        if self.coupling in ['axial','vector']:
            mmeds = np.linspace(100,3000,100)
        else:
            mmeds = np.linspace(0,1000,100)
        for mmed in mmeds:
            for mdmfrac in self.mdm_fractions_for_grid(mmed):
                x.append(mmed)
                y.append(mdmfrac * mmed)
                z.append(self.mu_2d(mmed, mmed * mdmfrac))
        ix,iy,iz = interpolate_rbf(x,y,z,maxval=max(mmeds))
        self.grid = (ix,iy,iz)

    def get_contours(self, level):
        ix, iy, iz = self.grid
        results = find_contours(iz, level=level)

        # Translation indices -> (x,y)
        fx = interp1d(
            list(range(0,len(ix[0]))),
            sorted(list(set(ix.flatten())))
            )
        fy = interp1d(
            list(range(0,len(iy[0]))),
            sorted(list(set(iy.flatten())))
            )

        contours = []
        for result in results:
            contours.append(
                            (fx(result[:,1]),fy(result[:,0]))
                            )

        return contours
