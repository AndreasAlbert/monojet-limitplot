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
        if new > 0:
            ret.append(reference/new)
        else:
            ret.append(np.nan)
    return ret

class DMInterp():
    def __init__(self, df, quantile='exp'):
        self.df = df
        self.quantile=quantile

        couplings = set(df.coupling)
        assert(len(couplings)==1)
        self.coupling = couplings.pop()

        f = interp1d(
                             df[df.mdm==1].mmed, 
                             np.log(df[df.mdm==1][quantile]), 
                             fill_value='extrapolate'
                             )
        self.mu_1d_mmed = lambda x: np.exp(f(x))
        self.calculate_grid()

    def mdm_correction_factor(self, mmed, mdm):
        '''Correction factor for going from mdm~0 to finite mdm'''
        mdm_threshold = 0.48
        factor = fdm_analytic(
                            mmed=itertools.repeat(mmed), 
                            mdm=[min(mdm, mdm_threshold*mmed)],
                            coupling=self.coupling
                            )[0]

        if mdm/mmed > 0.5:
            factor *= np.exp((mdm/(mmed*mdm_threshold))**8)
        return factor
    
    def mu_2d(self, mmed, mdm):
        mu = self.mu_1d_mmed(mmed)
        mu *= self.mdm_correction_factor(mmed, mdm)
        return mu

    def calculate_grid(self):
        x, y, z = [], [], []
        for mmed in np.linspace(100,3000,100):
            for mdmfrac in 0.6-np.logspace(np.log10(0.6),-2,30): #np.linspace(0,0.6,20):
                x.append(mmed)
                y.append(mdmfrac * mmed)
                z.append(self.mu_2d(mmed, mmed * mdmfrac))
        ix,iy,iz = interpolate_rbf(x,y,z,maxval=3000)
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
