import uproot
import itertools
import os
import pickle
from collections import defaultdict

import matplotlib.colors as mcolors
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, Rbf, interp1d
from scipy.optimize import minimize

from limitlib import (brazilgreen, brazilyellow, find_intersection,
                      interpolate_rbf)
from mediator_width import *
from dmsimp_lib import *
from dmcontour import DMInterp
pjoin = os.path.join
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

def plot_1d(df, tag):
    outdir = f'./output/{tag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
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
            fig.savefig(pjoin(outdir, f"{coupling}_1d.{ext}"))
            
        plt.close(fig)


def binned_fill(x, low, high, **kwargs):
    label = kwargs.pop("label")

    first = True
    width = 100
    for ix, il, ih in zip(x,low, high):
        plt.fill_between([ix-width,ix+width],[il,il],[ih,ih],label=label if first else None, **kwargs)
        first = False





def plot_coupling(df, tag, coupling_type='gq', correct_mdm=False):

    outdir = pjoin("./output/",tag)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    output_dfs = []
    for coupling in 'vector', 'axial':
        idf = df[(df.mdm==1)&(df.coupling==coupling)]
        if correct_mdm:
            corr_fac = fdm_analytic(idf.mmed, np.ones(len(idf.mmed))*1./3, coupling)
            for percentile in ['exp','p1s','p2s','m1s','m2s']:
                idf[percentile] = corr_fac * idf[percentile]

                
        fig = plt.gcf()
        hep.cms.cmslabel(data=True, year='2016-2018', lumi=137)
        ax = plt.gca()
        if coupling_type=='gq':
            exp = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.exp)])
            p1s = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.p1s)])
            m1s = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.m1s)])
            p2s = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.p2s)])
            m2s = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.m2s)])
        elif coupling_type=='gchi':
            exp = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.exp)])
            p1s = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.p1s)])
            m1s = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.m1s)])
            p2s = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.p2s)])
            m2s = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.m2s)])

        exp[np.isnan(exp) | np.isinf(exp)] = 1e2
        p1s[np.isnan(p1s) | np.isinf(p1s)] = 1e2
        p2s[np.isnan(p2s) | np.isinf(p2s)] = 1e2
        m1s[np.isnan(m1s) | np.isinf(m1s)] = 1e2
        m2s[np.isnan(m2s) | np.isinf(m2s)] = 1e2

        tmp_df = pd.DataFrame(
            {
            'exp' :exp,
            'p1s' : p1s,
            'p2s' : p2s,
            'm1s' : m1s,
            'm2s' : m2s,
            'mmed' : idf.mmed,
            'mdm' : np.ones(len(idf.mmed)) if not correct_mdm else idf.mmed/3,
            'coupling' : [coupling] * len(exp),
            'coupling_type' : [coupling_type]* len(exp),
            }
        )
        output_dfs.append(tmp_df)

        ax.plot(
                idf.mmed, 
                exp,
                marker='o',
                fillstyle='none',
                color='k',
                ls="--",
                label="Median expected",
                markersize=10,
                linewidth=2,
                zorder=2,
                )

        # binned_fill(idf.mmed, m1s, p1s,zorder=1,color=brazilgreen, label=r'68% expected')
        # binned_fill(idf.mmed, m2s, p2s,zorder=0,color=brazilyellow, label=r'95% expected')


        ax.fill_between(idf.mmed, m1s, p1s, color=brazilgreen,label=r'68% expected',zorder=1)
        ax.fill_between(idf.mmed, m2s, p2s, color=brazilyellow,label=r'95% expected',zorder=0)


        if coupling_type=='gq':
            ax.plot([0,2600],[0.25,0.25],lw=2,ls='-',color="crimson")
            ax.text(200,0.25,'$g_{q}$ = 0.25', color='crimson', va='bottom')
            ax.set_ylim(1e-2,0.5)
            ax.set_ylabel("Upper limit on the coupling $g_{q}$")
        elif coupling_type=='gchi':
            ax.plot([0,2600],[1.0,1.0],lw=2,ls='-',color="crimson")
            ax.text(200,1.0,'$g_{DM}$ = 1.0', color='crimson', va='bottom')
            ax.set_ylim(1e-2,2)
            ax.set_ylabel("Upper limit on the coupling $g_{DM}$")

        ax.set_yscale("log")
        ax.set_xlabel("$M_{med}$ (GeV)")
        ax.set_xlim(0,2600)
        ax.grid(ls='--')

        if correct_mdm:
            plt.legend(title=f'{coupling.capitalize()} mediator, $m_{{DM}}$ = $m_{{med}}$ / 3')
        else:
            plt.legend(title=f'{coupling.capitalize()} mediator, $m_{{DM}}$ = 1 GeV')
        for ext in ['png','pdf']:
            fig.savefig(pjoin(outdir, f"coupling_limit_{coupling}_{coupling_type}_1d_{'mdm1' if not correct_mdm else 'mdm_mmed_over_three'}.{ext}"))
            
        plt.close(fig)
        
    return output_dfs
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
    for mmed, mdm in zip(mmed, mdm):
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



def plot_2d(df, tag):

    outdir = pjoin("./output",tag)
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    for coupling in 'vector', 'axial':
        plt.gcf().clf()

        idf = df[df.coupling==coupling]
        dmc = DMInterp(idf)

        logz = True
        if(logz):
            contours_filled = np.log10(np.logspace(-1,1,7))
            contours_line = [0]
        else:
            contours_filled = [np.log10(0.1 * x) for x in range(1,50)]
            contours_line = [1]

        ix,iy,iz = dmc.grid
        if logz: 
            iz = np.log10(iz)

        fig = plt.figure(figsize=(14,10))

        iz[iz<min(contours_filled)] = min(contours_filled)
        iz[iz>max(contours_filled)] = max(contours_filled)
        CF = plt.contourf(ix, iy, iz, levels=contours_filled, cmap=cmap)
        for c in CF.collections:
            c.set_edgecolor("face")
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


        relic_contours = load_relic(coupling)
        for x,y in relic_contours:
            plt.plot(x,y, color='gray',lw=2)

        if coupling == 'axial':
            plt.text(1500,1000,"$\Omega h^2$ = 0.12", color="gray", rotation=40)
        else:
            plt.text(2400,600,"$\Omega h^2$ = 0.12", color="gray", rotation=30)
        for ext in 'pdf','png':
            plt.gcf().savefig(pjoin(outdir, f"{coupling}_contour.{ext}"))
        plt.close(plt.gcf())

def load_relic(coupling):
    with open(f"input/relic/relic_{coupling[0].capitalize()}1.pkl", "rb") as f:
        contours = pickle.load(f)
    return contours



def contour_to_dd(mmed, mdm, coupling, gq=0.25, gdm=1.0):
    '''Formulae from https://arxiv.org/pdf/1603.04156v1.pdf'''
    # Nucleon-DM reduced mass
    mn = 0.939
    mu = mn * mdm / (mn + mdm)
    # Type-dependent normalization
    if coupling=='vector':
        constant = 6.9e-41
    elif coupling=='axial':
        constant = 2.4e-42
    sigma_dd = constant * (gq*gdm/0.25)**2 * (1e3/mmed)**4 * mu**2
    return sigma_dd

def plot_dd_refs(coupling):


    colors = list(reversed([
        '#f7fbff',
        # '#deebf7',
        '#c6dbef',
        # '#9ecae1',
        '#6baed6',
        # '#4292c6',
        '#2171b5',
        # '#08519c',
        '#08306b'
    ]))
    if coupling == 'vector':
        results = {
            'Xenon1T 2018' : "input/dd/si/xenon1t_2018.txt",
            'Cresst-II': "input/dd/si/cresstii.txt",
            'CDMSlite': "input/dd/si/cdmslite2015.txt",
            'LUX' : "input/dd/si/LUX_SI_Combination_Oct2016.txt"
        }
    elif coupling == 'axial':
        results = {
            'Pico 2L' : "input/dd/sd/Pico2L.txt",
            'Pico60' : "input/dd/sd/Pico60.txt",
            'Picasso' : "input/dd/sd/PicassoFinal.root",
        }
    for i, (name, file) in enumerate(results.items()):
        if file.endswith("txt"):
            data = np.loadtxt(file)
            x = data[:,0]
            y = data[:,1]
        elif file.endswith("root"):
            f = uproot.open(file)
            x=f["Obs_90"].xvalues
            y=f["Obs_90"].yvalues
        plt.plot(x, y, label=name, color=colors[i], lw=3)

def plot_dd(df, tag):

    outdir = pjoin("./output",tag)
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    for coupling in 'vector', 'axial':
        plt.gcf().clf()

        idf = df[df.coupling==coupling]
        dmi = DMInterp(idf)
        contours = dmi.get_contours(level=1.0)
        assert(len(contours)==1)
        mmed, mdm = contours[0]

        mask = mmed > 250
        mmed = mmed[mask]
        mdm = mdm[mask]

        for fill in np.logspace(1,mdm[-1],10):
            mmed = np.r_[mmed, mmed[-1]]
            mdm = np.r_[mdm, fill]
        
        mask = mdm!=0
        mmed=mmed[mask]
        mdm=mdm[mask]
        xs = contour_to_dd(mmed, mdm, coupling)
        plt.plot(mdm, xs,'-', color='crimson', lw=3, label=f'{coupling.capitalize()} mediator, Dirac DM\n$g_{{q}}$=0.25, $g_{{DM}}$=1.0')
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(1e-47,1e-36)
        plt.xlim(1, 2e3)
        plt.xlabel("$m_{DM}$ (GeV)")
        plt.ylabel("$\sigma_{DM-nucleon}$ (cm$^2$)")
        plt.text(1.7e3,3e-47,"90% CL", ha='right', color='gray')
        hep.cms.cmslabel(data=True, year='2016-2018', lumi=137)
        plot_dd_refs(coupling)
        plt.legend()
        for ext in 'pdf','png':
            plt.gcf().savefig(pjoin(outdir, f"{coupling}_dd.{ext}"))


def main():
    # Input
    # infile = "input//limit_df.pkl"
    # infile = "input/2020-09-14/limit_df.pkl"
    tag = '2020-10-04_03Sep20v7'
    infile = f'../input/{tag}/limit_df.pkl'
    df  = pd.read_pickle(infile)

    # Vanilla plots
    # df95 = df[df.cl==0.95]
    # plot_2d(df95,tag=tag)
    # plot_1d(df95, tag)

    # # Coupling plot
    # dfs = []
    # for cp in ['gq','gchi']:
    #     for correct in True, False:
    #         dfs.extend(plot_coupling(df95, tag=tag,coupling_type=cp, correct_mdm=correct))

    # dfout = pd.concat(dfs)
    # dfout.to_pickle(
    #     pjoin('./output/',tag, 'coupling_limit_df.pkl')
    # )

    # DD
    df90 = df[df.cl==0.90]
    plot_dd(df90, tag=tag)
if __name__ == "__main__":
    main()
