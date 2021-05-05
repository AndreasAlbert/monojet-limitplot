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
        if new > 0:
            ret.append(reference/new)
        else:
            ret.append(np.nan)
    return ret
class OffshellGenerator():
    def __init__(self, coupling):
        with open(f"/home/albert/repos/monojet/2021-04-23_2016offshell/output/offshell_functions_{coupling}_2016.pkl","rb") as f:
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

        self.offshell = OffshellGenerator(self.coupling)
        # with open(f"output/offshell/offshell_shape_{self.coupling}_mmed100.pkl","rb") as f:
        #     self.f_offshell100 = pickle.load(f)
        # with open(f"output/offshell/offshell_shape_{self.c2oupling}_mmed250.pkl","rb") as f:
        #     self.f_offshell250 = pickle.load(f)

        x = df[df.mdm==1].mmed
        y=np.log(df[df.mdm==1][quantile])
        f = interp1d(
                    x,
                    y,
                    fill_value='extrapolate'
                    )
        self.mu_1d_mmed = lambda x: np.exp(f(x))
        self.calculate_grid()

    def mdm_correction_factor(self, mmed, mdm):
        '''Correction factor for going from mdm~0 to finite mdm'''
        mdm_threshold = 0.45
        factor = fdm_analytic(
                            mmed=itertools.repeat(mmed),
                            mdm=[min(mdm, mdm_threshold*mmed)],
                            coupling=self.coupling
                            )[0]


        if mdm/mmed > mdm_threshold:
            factor *= self.offshell.evaluate_correction(mmed, mdm)

            # # corr250 = np.exp(self.f_offshell250(min(mdm/mmed, 0.6))) / np.exp(self.f_offshell250(mdm_threshold))
            # corr100 = np.exp(self.f_offshell100(mdm/mmed)) / np.exp(self.f_offshell100(mdm_threshold))

            # # corr = corr100 + (corr250-corr100)/150 * (mmed-100)
            # # # corr = corr100 * np.exp(-(mmed/150))
            # # # if corr<1:
            # #     # corr = np.exp((2*mdm/mmed)**8)
            # # # if mmed>500:
            # # # corr *= np.exp((mdm/(mmed*mdm_threshold))**8)

            # #     a = corr100
            # #     b = np.log(corr250 / a)
            # #     corr = a * np.exp(b * (mmed-100)/150 )
            # # else:
            # #     if mmed < 175:
            # #         corr = corr100
            # #     else:
            # #         corr=corr250
            # # factor *=  corr
            # factor *= corr100
            # if mmed>100:
            #     if self.coupling=='vector':
            #         factor *= 1 + (mmed-100)/400
            #     elif self.coupling=='axial':
            #         factor *= 1 + 3*(mmed-100)/400
            # print(self.coupling, mmed, mdm, corr)
        return factor

    def mu_2d(self, mmed, mdm):
        mu = self.mu_1d_mmed(mmed)
        mu *= self.mdm_correction_factor(mmed, mdm)
        return mu

    def mdm_fractions_for_grid(self,mmed):
        mdmfracmax = max(5 - 4 * (mmed-100) / 400, 0.75)
        fractions =  list(np.linspace(0.,0.35,8))
        # fractions += list(
        #                 mdmfracmax - np.logspace(
        #                                          np.log10(mdmfracmax-0.35),
        #                                          -2,
        #                                          40)
        #                 )
        fractions += list(np.linspace(0.35,0.6,10))
        fractions += list(
                        mdmfracmax - np.linspace(
                                                 mdmfracmax - 0.6,
                                                 0,
                                                 40)
                        )
        return sorted(fractions)
    def calculate_grid(self):
        x, y, z = [], [], []
        for mmed in np.linspace(100,3000,100):
            for mdmfrac in self.mdm_fractions_for_grid(mmed):
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
