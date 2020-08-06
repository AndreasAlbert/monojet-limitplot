import itertools
import pickle
from collections import defaultdict
from mediator_width import *
import matplotlib.colors as mcolors
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, Rbf, interp1d
from scipy.optimize import minimize

from limitlib import interpolate_rbf, brazilgreen, brazilyellow, find_intersection

plt.style.use(hep.style.CMS)

cmap = mcolors.LinearSegmentedColormap.from_list("n", list(reversed([
    '#fff5f0',
    '#fee0d2',
    '#fcbba1',
    '#fc9272',
    '#fb6a4a',
    '#ef3b2c',
    '#cb181d',
    '#a50f15',
    '#67000d',
        ])))

def plot_1d(df):
    for coupling in 'vector', 'axial':
        idf = df[(df.mdm==1)&(df.coupling==coupling)]

        fig = plt.gcf()
        hep.cms.cmslabel(data=True, year='2016-2018', lumi=137)
        ax = plt.gca()
        ax.plot(idf.mmed, idf.exp,marker='o',fillstyle='none',color='k',ls="-", label="Median expected", markersize=10, linewidth=2,zorder=2)
        ax.fill_between(idf.mmed, idf.m1s,idf.p1s, color=brazilgreen,label=r'68% expected',zorder=1)
        ax.fill_between(idf.mmed, idf.m2s,idf.p2s, color=brazilyellow,label=r'95% expected',zorder=0)

        res = find_intersection(idf.mmed, idf.exp, 1)[0]
        ymin, ymax = 1e-3, 1e1
        plt.plot([min(idf.mmed), max(idf.mmed)],[1,1],color='red',lw=2)
        plt.plot([res,res],[ymin, ymax],color='red',lw=2)
        ax.set_yscale("log")
        ax.set_xlabel("$M_{med}$ (GeV)")
        ax.set_ylabel("Upper limit on the signal strength $\mu$")
        ax.set_ylim(ymin, ymax)
        plt.legend(title=f'{coupling.capitalize()} mediator, $m_{{DM}}$ = 1 GeV')
        for ext in ['png','pdf']:
            fig.savefig(f"output/combination/{coupling}_1d.{ext}")
            
        plt.close(fig)

# def plot_coupling(df):
#     for coupling in 'vector', 'axial':
#         plt.gcf().clf()

#         idf = df[df.coupling==coupling]

#         idf1 = idf[idf.mdm==1]
#         fmed = interp1d(idf1.mmed, np.log(idf1.exp))
#         # mmed = np.linspace(0,2500,100)

#         idf2k = idf[idf.mmed==2000]
#         fdm = interp1d(idf2k.mdm / 2000, np.log(idf2k.exp / idf2k.exp[idf2k.mdm==1]), fill_value='extrapolate')

#         gamma_axial_chi(m_med, m_x, g_chi)gamma_axial_total()

def fdm_analytic(mmed, mdm, coupling):
    ret = []
    for mmed, mdm in zip(itertools.repeat(mmed), mdm):
        if coupling=='vector':
            reference = gamma_vector_chi(mmed, m_x=0, g_chi=1.0) / gamma_vector_total(mmed, m_x=0, g_chi=1.0, g_q=0.25)
            new = gamma_vector_chi(mmed, m_x=mdm*mmed, g_chi=1.0) / gamma_vector_total(mmed, m_x=mdm*mmed, g_chi=1.0, g_q=0.25)
        elif coupling == 'axial':
            reference = gamma_axial_chi(mmed, m_x=0, g_chi=1.0) / gamma_axial_total(mmed, m_x=0, g_chi=1.0, g_q=0.25)
            new = gamma_axial_chi(mmed, m_x=mdm*mmed, g_chi=1.0) / gamma_axial_total(mmed, m_x=mdm*mmed, g_chi=1.0, g_q=0.25)
        if new > 0:
            ret.append(reference/new)
        else:
            ret.append(np.nan)
    return ret



def plot_2d(df):
    for coupling in 'vector', 'axial':
        plt.gcf().clf()

        idf = df[df.coupling==coupling]

        idf1 = idf[idf.mdm==1]
        fmed = interp1d(idf1.mmed, np.log(idf1.exp))
        # mmed = np.linspace(0,2500,100)

        idf2k = idf[idf.mmed==2000]
        fdm = interp1d(idf2k.mdm / 2000, np.log(idf2k.exp / idf2k.exp[idf2k.mdm==1]), fill_value='extrapolate')
        
        plt.gcf().clf()
        mdm = np.linspace(0,0.5,100)
        plt.plot(idf2k.mdm / 2000, idf2k.exp / idf2k.exp[idf2k.mdm==1],marker='o',markersize=10, label='Reco-level result',ls='none', color='k')
        plt.plot(mdm, np.exp(fdm(mdm)),lw=2,label='Log interpolation of reco',color='k',ls='--')
        plt.plot(mdm, fdm_analytic(mmed=2000, mdm=mdm, coupling=coupling), lw=2,label='Analytic BR scaling', color='r')

        plt.xlabel("$m_{DM}$ / $m_{med}$")
        plt.ylabel("$\mu(m_{DM})$ / $\mu(m_{DM}$ = 1 GeV)")
        plt.ylim(0,20 if coupling=='axial' else 5)
        plt.legend(title=f'{coupling.capitalize()} mediator')

        for ext in 'pdf','png':
            plt.gcf().savefig(f"output/combination/{coupling}_fdm.{ext}")
        plt.gcf().clf()

        x, y, z = [], [], []
        for mmed in np.linspace(100,2500,100):
            for mdm in np.linspace(0,0.6,10):
                if mdm < 0.5:
                    mu = np.exp(fmed(mmed)) * fdm_analytic(mmed=mmed, mdm=[mdm],coupling=coupling)[0]
                else:
                    mu = np.exp(fmed(mmed)) * fdm_analytic(mmed=mmed, mdm=[0.48],coupling=coupling)[0] * np.exp((mdm/0.48)**8)
                x.append(mmed)
                y.append(mdm*mmed)
                z.append(mu)
        

        logz = True
        if(logz):
            contours_filled = np.log10(np.logspace(-1,1,7))
            contours_line = [0]
        else:
            contours_filled = [np.log10(0.1 * x) for x in range(1,50)]
            contours_line = [1]


        fig = plt.figure(figsize=(14,10))
        ix,iy,iz = interpolate_rbf(x,y,z,maxval=3000)
        if logz: iz = np.log10(iz)
        iz[iz<min(contours_filled)] = min(contours_filled)
        iz[iz>max(contours_filled)] = max(contours_filled)
        CF = plt.contourf(ix, iy, iz, levels=contours_filled, cmap=cmap)
        cb = plt.colorbar()
        CS2 = plt.contour(ix, iy, iz, levels=contours_line, colors="navy", linestyles="solid",linewidths = 3, zorder=2)
        cb.add_lines(CS2)

        hep.cms.cmslabel(data=True, year='2016-2018', lumi=137)
        plt.clim([1e-1,1e1])
        cb.set_label("95% CL expected limit on $\log_{10}(\mu)$")
        plt.plot([0,3000],[0,1500],'--',color='gray')
        plt.xlabel("$m_{med}$ (GeV)")
        plt.ylabel("$m_{DM} $(GeV)")
        plt.ylim(0,1500)
        plt.xlim(0,3000)
        plt.text(50,1300, 
        '\n'.join([
            f'{coupling.capitalize()} mediator',
            '$g_{q} = 0.25, g_{\chi} = 1.0$'
        ])
        )
        for ext in 'pdf','png':
            plt.gcf().savefig(f"output/combination/{coupling}_contour.{ext}")
        plt.close(plt.gcf())


def main():
    df  = pd.read_pickle("input/limit_df.pkl")
    # print(df.to_string())
    plot_2d(df)
if __name__ == "__main__":
    main()
