import os
import pickle
import re

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from limitlib import fill_dummy_values, interpolate_rbf, load_directory
import mplhep as hep
import os
from scipy import interpolate
pjoin = os.path.join
plt.style.use(hep.style.CMS)

def find_intersection(x,y,value=1):

    # Find intersection
    # try:
    xinterp = np.linspace(0, 10*max(x), 1000)
    finterp = interpolate.interp1d(x, y,kind='linear' if len(x)<4 else 'quadratic',fill_value="extrapolate", bounds_error=False)
    yinterp = finterp(xinterp)

    idx = np.argwhere(np.diff(np.sign(yinterp - value))).flatten()
    if not len(idx):
        if np.all(y<value):
            return 1e-6
        else:
            return -1
    else:
        ret = xinterp[idx[0]] 

        if ret < 0.5*min(x) or ret > 3*max(x):
            return -1
        else:
            return ret

def plot_1d(limits, tag):
    outdir = f'./output/{tag}/coupling'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    limits.sort_values(by='ylq', inplace=True)


    mlq_list = []
    ylq_exp_list = []
    ylq_obs_list = []
    ylq_p1s_list = []
    ylq_p2s_list = []
    ylq_m1s_list = []
    ylq_m2s_list = []
    for mlq in set(limits.mlq):
        if mlq > 2000:
            continue
        print(mlq)
        ilims = limits[limits.mlq==mlq]

        plt.fill_between(ilims.ylq,ilims.m2s,ilims.p2s,color='orange', label='Expected $\pm$ 2 s.d.')
        plt.fill_between(ilims.ylq,ilims.m1s,ilims.p1s,color='green', label='Expected $\pm$ 1 s.d.')
        plt.plot(ilims.ylq,ilims.exp,'k--o', label='Expected', fillstyle='none')
        plt.plot(ilims.ylq,ilims.obs,'k-o', label='Observed')
        plt.yscale("log")
        # plt.xscale("log")
        plt.plot([min(ilims.ylq), max(ilims.ylq)],[1,1],'r-')

        ymin = 1e-3
        plt.ylim(ymin,20)
        plt.xlabel("$y_{LQ}$")
        plt.ylabel("95% exclusion limits on $\mu$")
        plt.legend()
        hep.cms.label(data=True, year='2016-2018', lumi=137, paper=True)

        # plt.xlim(0,1000)
        


        ylq_obs = find_intersection(ilims.ylq, ilims.obs)
        ylq_exp = find_intersection(ilims.ylq, ilims.exp)
        ylq_m1s = find_intersection(ilims.ylq, ilims.m1s)
        ylq_m2s = find_intersection(ilims.ylq, ilims.m2s)
        ylq_p1s = find_intersection(ilims.ylq, ilims.p1s)
        ylq_p2s = find_intersection(ilims.ylq, ilims.p2s)

        if ylq_obs > 0:
            plt.plot([ylq_obs, ylq_obs],[ymin,1e1],'r-')
        if ylq_exp > 0:
            plt.plot([ylq_exp, ylq_exp],[ymin,1e1],'r-')
        if ylq_p1s > 0:
            plt.plot([ylq_p1s, ylq_p1s],[ymin,1e1],'r--')
        if ylq_m1s > 0:
            plt.plot([ylq_m1s, ylq_m1s],[ymin,1e1],'r--')

        ylq_obs_list.append(ylq_obs)
        ylq_exp_list.append(ylq_exp)
        ylq_p1s_list.append(ylq_p1s)
        ylq_m1s_list.append(ylq_m1s)
        ylq_p2s_list.append(ylq_p2s)
        ylq_m2s_list.append(ylq_m2s)
        mlq_list.append(mlq)
        plt.plot([min(ilims.ylq), max(ilims.ylq)],[1,1],'r-')

        text = [
            f'Scalar first-generation leptoquark',
            f'$m_{{LQ}} = {mlq}$ GeV'
        ]
        if ylq_exp > 0:
            text.append(f'Coupling limit = {ylq_exp:.2f} ({ylq_p1s:.2f} -- {ylq_m1s:.2f})')
        plt.text(1.1*min(ilims.ylq),0.05, 
        '\n'.join(text)
        )

        for extension in ['pdf','png']:
            plt.gcf().savefig(pjoin(outdir, f"mlq_{mlq}_coupling.{extension}"))
        plt.gcf().clf()

    mlq_list = np.array(mlq_list)
    ylq_p1s_list = np.array(ylq_p1s_list)
    ylq_p2s_list = np.array(ylq_p2s_list)
    ylq_m1s_list = np.array(ylq_m1s_list)
    ylq_m2s_list = np.array(ylq_m2s_list)
    ylq_exp_list = np.array(ylq_exp_list)
    ylq_obs_list = np.array(ylq_obs_list)

    sorter = mlq_list.argsort()
    mlq_list = mlq_list[sorter]
    ylq_exp_list = ylq_exp_list[sorter]
    ylq_obs_list = ylq_obs_list[sorter]
    ylq_p1s_list = ylq_p1s_list[sorter]
    ylq_p2s_list = ylq_p2s_list[sorter]
    ylq_m1s_list = ylq_m1s_list[sorter]
    ylq_m2s_list = ylq_m2s_list[sorter]

    mask = (ylq_exp_list>0) &(mlq_list<2250)
    plt.plot(mlq_list[mask],ylq_exp_list[mask],'k--o', label='Median expected', fillstyle='none', ms=10,lw=2)
    plt.plot(mlq_list[mask],ylq_obs_list[mask],'k-o', label='Observed', ms=10,lw=2)

    # plt.fill_between(mlq_list[mask],ylq_m2s_list[mask],ylq_p2s_list[mask],color='orange', label='Expected $\pm$ 2 s.d.',zorder=-1)
    plt.fill_between(mlq_list[mask],ylq_m1s_list[mask],ylq_p1s_list[mask],color='green', label='Median expected $\pm$ 1 s.d.',zorder=0)
    plt.fill_between(mlq_list[mask],ylq_m2s_list[mask],ylq_p2s_list[mask],color='orange', label='Median expected $\pm$ 2 s.d.',zorder=-1)
    plt.xlabel("Leptoquark mass (GeV)")
    plt.ylabel("95% CL upper limit on $\lambda$")
    hep.cms.label(data=True, year='2016-2018', lumi=137, paper=True)



    text = [
            f'Scalar first-generation\nleptoquark',
        ]
    plt.text(500,1.4, 
        '\n'.join(text)
        )
    plt.legend(loc='lower right')
    for extension in ['pdf','png']:
        plt.gcf().savefig(pjoin(outdir, f"mlq_coupling.{extension}"))
    plt.gcf().clf()

def main():

    tag = '2021-03-25_unblind_2021-03-27_unblind_v2_default_templatereplace_v9'
    df = pd.read_pickle(f"../input/{tag}/limit_df.pkl")
    df = df[(df.cl==0.95)& (~np.isnan(df.mlq)) &  (~np.isnan(df.ylq))]
    # limits = pd.concat(
    #     [load_directory("./input/scan_lq")]
    # )

    # print(limits)
    # # print(load_directory("./input/scan_806"))
    # limits.drop_duplicates(subset=['mlq','ylq','tag'],keep="first",inplace=True)
    # print(limits)
    plot_1d(df,tag)
    # limits = load("./input/2020-06-08")
    # print(limits)
    # contour_plot(limits, outdir=pjoin("./output", indir.split('/')[-1]))
    # contour_plot(limits, outdir=pjoin("./output", 'merge'))
if __name__ == "__main__":
    main()
