import uproot
import os
import pickle

import matplotlib.colors as mcolors
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from limitlib import (brazilgreen, brazilyellow, find_intersection,
                      dump_contour_to_txt)
from mediator_width import *
from dmsimp_lib import *
from dmcontour import DMInterp
pjoin = os.path.join
plt.style.use(hep.style.CMS)

cmap = mcolors.LinearSegmentedColormap.from_list("n", list(reversed([
    # '#fff5f0',
    '#fee0d2',
    '#ffffff',
    '#fcbba1',
    '#fc9272',
    '#fb6a4a',
    '#ef3b2c',
    '#cb181d',
    '#a50f15',
    '#67000d',
    # '#000000',
        ])))

def plot_1d(df, tag):
    outdir = f'./output/{tag}/spin1'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for coupling in 'vector', 'axial':
        idf = df[(df.mdm==1)&(df.coupling==coupling)]

        fig = plt.gcf()
        hep.cms.label(data=True, year='2016-2018', lumi=137)
        ax = plt.gca()
        ax.plot(idf.mmed, idf.exp,marker='o',fillstyle='none',color='k',ls="-", label="Median expected", markersize=10, linewidth=2,zorder=2)
        ax.fill_between(idf.mmed, idf.m1s,idf.p1s, color=brazilgreen,label=r'68% expected',zorder=1)
        ax.fill_between(idf.mmed, idf.m2s,idf.p2s, color=brazilyellow,label=r'95% expected',zorder=0)

        res = find_intersection(idf.mmed, idf.exp, 1)[0]
        ymin, ymax = 1e-3, 1e1
        plt.plot([min(idf.mmed), max(idf.mmed)],[1,1],color='red',lw=2)
        plt.plot([res,res],[ymin, ymax],color='red',lw=2)
        ax.set_yscale("log")
        ax.set_xlabel("$m_{med}$ (GeV)")
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


def get_relic_coupling(coupling, mediator):
    if coupling=='gq':
        if mediator=='axial':
            fname = 'gqa.txt'
        elif mediator=='vector':
            fname = 'gqv.txt'
    if coupling=='gchi':
        if mediator=='axial':
            fname = 'gdma.txt'
        elif mediator=='vector':
            fname = 'gdmv.txt'
    data = np.loadtxt(f'input/relic/{fname}')
    x = data[:,0]
    y = data[:,1]
    return x, y 

def relic_labels(mediator, coupling):
    text = "$\Omega h^2$ = 0.12"
    color = "mediumblue"
    fontsize=24
    if mediator=='axial':
        if coupling=='gq':
            plt.text(600,0.11, text, color=color, rotation=40,fontsize=fontsize)
        elif coupling=='gchi':
            plt.text(250,0.2, text, color=color, rotation=50,fontsize=fontsize)
        else:
            raise RuntimeError
    elif mediator=='vector':
        if coupling=='gq':
            plt.text(1900,0.06, text, color=color, rotation=12,fontsize=fontsize)
        elif coupling=='gchi':
            plt.text(1900,0.23, text, color=color, rotation=12,fontsize=fontsize)
        else:
            raise RuntimeError
    else:
        raise RuntimeError

def plot_coupling(df, tag, coupling_type='gq', correct_mdm=False):

    outdir = pjoin("./output/",tag,'spin1')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    output_dfs = []
    for coupling in 'vector', 'axial':
        idf = df[(df.mdm==1)&(df.coupling==coupling)]
        idf = idf.sort_values(by='mmed')
        if correct_mdm:
            corr_fac = fdm_analytic(idf.mmed, np.ones(len(idf.mmed))*1./3, coupling)
            for percentile in ['obs','exp','p1s','p2s','m1s','m2s']:
                idf[percentile] = corr_fac * idf[percentile]


        fig = plt.gcf()

        ax = plt.gca()
        if coupling_type=='gq':
            obs = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.obs)])
            exp = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.exp)])
            p1s = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.p1s)])
            m1s = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.m1s)])
            p2s = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.p2s)])
            m2s = np.array([determine_gq_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.m2s)])
        elif coupling_type=='gchi':
            obs = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.obs)])
            exp = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.exp)])
            p1s = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.p1s)])
            m1s = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.m1s)])
            p2s = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.p2s)])
            m2s = np.array([determine_gchi_limit_analytical(mediator=coupling, mmed=m, mdm=1, mu=mu, gq_reference=0.25, gchi_reference=1.0) for m,mu in zip(idf.mmed, idf.m2s)])

        obs[np.isnan(obs) | np.isinf(obs)] = 1e9
        exp[np.isnan(exp) | np.isinf(exp)] = 1e9
        p1s[np.isnan(p1s) | np.isinf(p1s)] = 1e9
        p2s[np.isnan(p2s) | np.isinf(p2s)] = 1e9
        m1s[np.isnan(m1s) | np.isinf(m1s)] = 1e9
        m2s[np.isnan(m2s) | np.isinf(m2s)] = 1e9

        tmp_df = pd.DataFrame(
            {
            'obs' :obs,
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

        mask = (obs<1e1) & (exp<1e1) & (p2s<1e1) & (m2s<1e1)
        mask = obs>0
        ax.plot(
                idf.mmed[mask],
                exp[mask],
                marker='o',
                fillstyle='none',
                color='k',
                ls="--",
                label="Median expected",
                markersize=10,
                linewidth=2,
                zorder=2,
                )
        ax.plot(
                idf.mmed[mask],
                obs[mask],
                marker='o',
                color='k',
                ls="-",
                label="Observed",
                markersize=10,
                linewidth=2,
                zorder=2,
                )

        # binned_fill(idf.mmed, m1s, p1s,zorder=1,color=brazilgreen, label=r'68% expected')
        # binned_fill(idf.mmed, m2s, p2s,zorder=0,color=brazilyellow, label=r'95% expected')


        ax.fill_between(idf.mmed[mask], m1s[mask], p1s[mask], color=brazilgreen,label=r'68% expected',zorder=1)
        ax.fill_between(idf.mmed[mask], m2s[mask], p2s[mask], color=brazilyellow,label=r'95% expected',zorder=0)


        if coupling_type=='gq':
            ax.plot([0,2600],[0.25,0.25],lw=2,ls='-',color="crimson")
            ax.text(200,0.23,'$g_{q}$ = 0.25', color='crimson', va='top')
            ax.set_ylim(1e-2,0.5)
            ax.set_ylabel("95% CL upper limit on the coupling $g_{q}$")
        elif coupling_type=='gchi':
            ax.plot([0,2600],[1.0,1.0],lw=2,ls='-',color="crimson")
            ax.text(200,0.9,'$g_{DM}$ = 1.0', color='crimson', va='top')
            ax.set_ylim(1e-2,2)
            ax.set_ylabel("95% CL upper limit on the coupling $g_{\chi}$")

        ax.set_yscale("log")
        ax.set_xlabel("$m_{med}$ (GeV)")
        ax.set_xlim(0,2500)
        ax.grid(ls='--')

        if coupling_type == 'gq':
            coupling_statement = '$g_{\chi} = 1.0$'
        else:
            coupling_statement = '$g_{q} = 0.25$'
        if correct_mdm:
            mdm_statement = '$m_{{DM}}$ = $m_{{med}}$ / 3'
        else:
            mdm_statement =  '$m_{{DM}}$ = 1 GeV'

        if correct_mdm:
            x_relic, y_relic = get_relic_coupling(coupling=coupling_type, mediator=coupling)
            relic_labels(mediator=coupling, coupling=coupling_type)
            ax.plot(x_relic,y_relic, color='mediumblue',lw=2)

        if coupling_type=='gq':
            if coupling=='vector':
                plt.text(100,0.15, f'{coupling.capitalize()} mediator\n{coupling_statement}\n{mdm_statement}', ha='left',va='top')
                plt.legend()
            elif coupling=='axial':
                plt.text(2450,0.08, f'{coupling.capitalize()} mediator\n{coupling_statement}\n{mdm_statement}', ha='right',va='top')
                plt.legend()
        elif coupling_type=='gchi':
            plt.text(2400,0.035, f'{coupling.capitalize()} mediator\n{coupling_statement}\n{mdm_statement}', ha='right',va='top')
            plt.legend(loc=(0.1,1.5e-2))
        hep.cms.label(data=True, year='2016-2018', lumi=137,loc=1)
        for ext in ['png','pdf']:
            fig.savefig(pjoin(outdir, f"coupling_limit_{coupling}_{coupling_type}_1d_{'mdm1' if not correct_mdm else 'mdm_mmed_over_three'}.{ext}"))

        labels = hep.cms.label(data=True, year='2016-2018', lumi=137, label='Supplementary',loc=1)
        for ext in ['png','pdf']:
            fig.savefig(pjoin(outdir, f"coupling_limit_{coupling}_{coupling_type}_1d_{'mdm1' if not correct_mdm else 'mdm_mmed_over_three'}_supplementary.{ext}"))

        labels[1].remove()
        hep.cms.label(data=True, label="Preliminary", year='2016-2018', lumi=137, loc=1)
        for ext in ['png','pdf']:
            fig.savefig(pjoin(outdir, f"coupling_limit_{coupling}_{coupling_type}_1d_{'mdm1' if not correct_mdm else 'mdm_mmed_over_three'}_preliminary.{ext}"))
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

    outdir = pjoin("./output/",tag,'spin1')
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    for coupling in 'vector', 'axial':
        plt.gcf().clf()

        idf = df[df.coupling==coupling]
        dmc = DMInterp(idf)
        dmc_p1s = DMInterp(idf, quantile='p1s')
        dmc_m1s = DMInterp(idf, quantile='m1s')
        dmc_p2s = DMInterp(idf, quantile='p2s')
        dmc_m2s = DMInterp(idf, quantile='m2s')
        dmc_obs = DMInterp(idf, quantile='obs')

        logz = True
        if(logz):
            contours_filled = np.log10(np.logspace(-1,1,7))
            contours_line = [0]
        else:
            contours_filled = [np.log10(0.1 * x) for x in range(1,50)]
            contours_line = [1]

        ix,iy,iz = dmc.grid
        ix_p1s,iy_p1s,iz_p1s = dmc_p1s.grid
        ix_m1s,iy_m1s,iz_m1s = dmc_m1s.grid
        ix_p2s,iy_p2s,iz_p2s = dmc_p2s.grid
        ix_m2s,iy_m2s,iz_m2s = dmc_m2s.grid
        ix_obs,iy_obs,iz_obs = dmc_obs.grid
        if logz:
            iz = np.log10(iz)
            iz_p1s = np.log10(iz_p1s)
            iz_m1s = np.log10(iz_m1s)
            iz_p2s = np.log10(iz_p2s)
            iz_m2s = np.log10(iz_m2s)
            iz_obs = np.log10(iz_obs)

        fig = plt.figure(figsize=(14,10))

        iz[iz<min(contours_filled)] = min(contours_filled)
        iz[iz>max(contours_filled)] = max(contours_filled)

        iz_obs[iz_obs<min(contours_filled)] = min(contours_filled)
        iz_obs[iz_obs>max(contours_filled)] = max(contours_filled)

        iz_obs[iz_obs<min(contours_filled)] = min(contours_filled)
        iz_obs[iz_obs>max(contours_filled)] = max(contours_filled)

        CF = plt.contourf(ix_obs, iy_obs, iz_obs, levels=contours_filled, cmap=cmap,zorder=-5,alpha=0.85)
        for c in CF.collections:
            c.set_edgecolor("none")
        cb = plt.colorbar()

        color_obs = 'blue'
        color_exp = 'navy'

        contours = {}
        contour_exp = plt.contour(ix, iy, iz, levels=contours_line, colors=color_exp, linestyles=[(0, (5,1))],linewidths = 2, zorder=2)
        contour_exp.collections[0].set_label('Median expected')
        contour_p1s = plt.contour(ix_p1s, iy_p1s, iz_p1s, levels=contours_line, colors=color_exp, linestyles=[(0, (3,3))],linewidths = 2, zorder=2)
        contour_p1s.collections[0].set_label(r'68% Expected')
        contour_m1s = plt.contour(ix_m1s, iy_m1s, iz_m1s, levels=contours_line, colors=color_exp, linestyles=[(0, (3,3))],linewidths = 2, zorder=2)
        
        contour_p2s = plt.contour(ix_p2s, iy_p2s, iz_p2s, levels=contours_line, colors=color_exp, linestyles=[(0, (1,5))],linewidths = 2, zorder=2)
        contour_p2s.collections[0].set_label(r'95% Expected')
        contour_m2s = plt.contour(ix_m2s, iy_m2s, iz_m2s, levels=contours_line, colors=color_exp, linestyles=[(0, (1,5))],linewidths = 2, zorder=2)

        contour_obs = plt.contour(ix_obs, iy_obs, iz_obs, levels=contours_line, colors=color_obs, linestyles="solid",linewidths = 4, zorder=2)
        contour_obs.collections[0].set_label('Observed')
        cb.add_lines(contour_obs)

        plt.clim([1e-1,1e1])
        cb.set_label("95% CL upper observed limit on $\log_{10}(\mu)$")
        plt.plot([0,3000],[0,1500],'--',color='gray')
        plt.xlabel("$m_{med}$ (GeV)")
        plt.ylabel("$m_{DM}$ (GeV)")
        plt.ylim(0,1500)
        plt.xlim(0,3000)
        # plt.text(100,1300,
        # '\n'.join([
        #     f'{coupling.capitalize()} mediator',
        #     '$g_{q} = 0.25, g_{\chi} = 1.0$'
        # ])
        # )
        draw_2016(coupling)

        plt.legend(loc='upper left', title= '\n'.join([
            f'{coupling.capitalize()} mediator',
            '$g_{q} = 0.25, g_{\chi} = 1.0$'
        ]),frameon=True,edgecolor='none',framealpha=0.75)
        relic_contours = load_relic(coupling)
        for x,y in relic_contours:
            plt.plot(x,y, color='gray',lw=2)

        if coupling == 'axial':
            plt.text(1500,1000,"$\Omega h^2$ = 0.12", color="gray", rotation=40)
        else:
            plt.text(2400,600,"$\Omega h^2$ = 0.12", color="gray", rotation=30)
        
        
        labels = hep.cms.label(data=True, year='2016-2018', lumi=137)
        for ext in 'pdf','png':
            plt.gcf().savefig(pjoin(outdir, f"{coupling}_contour.{ext}"))

        labels[1].remove()
        hep.cms.label(data=True, label="Preliminary", year='2016-2018', lumi=137)
        for ext in 'pdf','png':
            plt.gcf().savefig(pjoin(outdir, f"{coupling}_contour_preliminary.{ext}"))

                    
        # if coupling == 'axial':
        #     draw_atlas(coupling)
        #     plt.legend(loc='upper left')
        #     for ext in 'pdf','png':
        #         plt.gcf().savefig(pjoin(outdir, f"{coupling}_contour_withatlas.{ext}"))
        plt.close(plt.gcf())

        dump_contour_to_txt(contour_exp, pjoin(outdir, f"contour_{coupling}_exp.txt"))
        dump_contour_to_txt(contour_obs, pjoin(outdir, f"contour_{coupling}_obs.txt"))
        dump_contour_to_txt(contour_p1s, pjoin(outdir, f"contour_{coupling}_p1s.txt"))
        dump_contour_to_txt(contour_m1s, pjoin(outdir, f"contour_{coupling}_m1s.txt"))

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
        # '#f7fbff',
        # '#deebf7',
        '#c6dbef',
        '#9ecae1',
        '#6baed6',
        # '#4292c6',
        '#2171b5',
        # '#08519c',
        '#08306b',
        'k',
        'b'
    ]))
    if coupling == 'vector':
        results = {
            'Xenon1T 2018' : "input/dd/si/xenon1t_2018.txt",
            'Cresst-II': "input/dd/si/cresstii.txt",
            'CDMSlite': "input/dd/si/cdmslite2015.txt",
            'LUX' : "input/dd/si/LUX_SI_Combination_Oct2016.txt",
            'Panda-X II' : "input/dd/si/pandax_132_tonday_rescale.txt",
            # 'Cresst-III' : "input/dd/si/cresstiii_2019.txt",
            'DarkSide-50' : "input/dd/si/darkside.txt",
        }
    elif coupling == 'axial':
        results = {
            'Pico 2L' : "input/dd/sd/Pico2L.txt",
            'Pico-60' : "input/dd/sd/Pico60.txt",
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

    outdir = pjoin("./output/",tag, "spin1")
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    for coupling in 'vector', 'axial':
        plt.gcf().clf()
        fig = plt.figure(figsize=(11,9))

        idf = df[df.coupling==coupling]
        dmi = DMInterp(idf, quantile='obs')
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
        plt.ylim(1e-47,1e-35)
        plt.xlim(1, 2e3)
        plt.xlabel("$m_{DM}$ (GeV)")
        plt.ylabel("Upper limit on $\sigma_{DM-nucleon}$ (cm$^2$)")
        plt.text(1.7e3,3e-47,"90% CL", ha='right', color='gray')

        if coupling=='axial':
            plt.text(1e3,1e-36,"Spin dependent", ha='right', color='gray')
        else:
            plt.text(1.1,3e-47,"Spin independent", ha='left', color='gray')


        # hep.cms.label(data=True, year=False, lumi=137, paper=True, supplementary=True)
        plot_dd_refs(coupling)
        if coupling=='axial':
            plt.legend(ncol=2)
        else:
            plt.legend(ncol=2)

        labels = hep.cms.label(data=True, label='Supplementary', lumi=137)
        for ext in 'pdf','png':
            plt.gcf().savefig(pjoin(outdir, f"{coupling}_dd_supplementary.{ext}"))

        labels[1].remove()
        hep.cms.label(data=True, label='Preliminary', lumi=137)
        for ext in 'pdf','png':
            plt.gcf().savefig(pjoin(outdir, f"{coupling}_dd_preliminary.{ext}"))

def draw_2016(mediator):
    f = uproot.open("input/2016/HEPData-ins1641762-v1-root.root")
    if mediator=='vector':
        gobs = f['Observed exclusion contour for vector mediator/Graph1D_y1']
        gexp = f['Expected exclusion contour for vector mediator/Graph1D_y1']
    elif mediator=='axial':
        gobs = f['Observed exclusion contour for axial-vector mediator/Graph1D_y1']
        gexp = f['Expected exclusion contour for axial-vector mediator/Graph1D_y1']

    color='gold'
    plt.plot(gobs.xvalues, gobs.yvalues, color=color, lw=2, label='2016 observed',zorder=-1)
    plt.plot(gexp.xvalues, gexp.yvalues, color=color, lw=2, ls=(0, (5,1)), label='2016 median expected',zorder=-1)

def draw_atlas(coupling):
    if coupling=='axial':
        points = [(14.104372355430087, 109.45529290853051),
        (155.14809590973198, 129.49640287769807),
        (273.6248236953455, 163.41212744090467),
        (400.5641748942171, 212.74409044193226),
        (530.324400564175, 262.0760534429603),
        (640.3385049365304, 309.8663926002057),
        (803.9492242595204, 380.78108941418304),
        (928.0677009873059, 430.11305241521086),
        (1060.6488011283498, 479.44501541623845),
        (1173.4837799717911, 511.8191161356631),
        (1320.1692524682649, 558.0678314491265),
        (1486.600846262341, 607.3997944501543),
        (1545.8392101551478, 615.1079136690648),
        (1658.6741889985894, 613.5662898252828),
        (1743.3004231311704, 613.5662898252828),
        (1802.5387870239772, 601.2332990750259),
        (1853.314527503526, 575.0256937307299),
        (1904.0902679830745, 550.359712230216),
        (1935.119887165021, 519.5272353545736),
        (1971.7912552891396, 477.9033915724565),
        (2000.0000000000002, 442.4460431654679),
        (2022.566995768688, 405.44707091469695),
        (2039.4922425952043, 371.53134635149036),
        (2078.984485190409, 303.69989722507717),
        (2101.5514809590977, 226.61870503597152),
        (2112.8349788434416, 171.12024665981517),
        (2121.297602256699, 120.24665981500539),
        (2126.939351198872, 64.74820143884926),
        (2132.581100141044, 3.0832476875643806)]
    else:
        return
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    plt.plot(x,y,color='magenta',lw=2,ls='--',label='ATLAS Run-2 expected')
def main():
    # Input
    # infile = "input//limit_df.pkl"
    # infile = "input/2020-09-14/limit_df.pkl"
    tag = '2021-05-03_master_default_default'
    infile = f'../input/{tag}/limit_df.pkl'
    df  = pd.read_pickle(infile)

    df.exp = 0.01 * df.exp
    df.obs = 0.01 * df.obs
    df.p1s = 0.01 * df.p1s
    df.p2s = 0.01 * df.p2s
    df.m1s = 0.01 * df.m1s
    df.m2s = 0.01 * df.m2s

    # Vanilla plots
    df95 = df[df.cl==0.95]
    plot_2d(df95,tag=tag)
    # # plot_1d(df95, tag)

    # Coupling plot
    dfs = []
    for cp in ['gq','gchi']:
        for correct in True, False:
            dfs.extend(plot_coupling(df95, tag=tag,coupling_type=cp, correct_mdm=correct))

    # dfout = pd.concat(dfs)
    # dfout.to_pickle(
    #     pjoin('./output/',tag, 'spin1/coupling_limit_df.pkl')
    # )


    # df90 = df[df.cl==0.90]
    # plot_dd(df90, tag=tag)
if __name__ == "__main__":
    main()
