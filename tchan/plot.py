import os
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from limitlib import fill_dummy_values, interpolate_rbf, load_directory

import pandas as pd

pjoin = os.path.join

import mplhep as hep

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


x16 = np.array([
 108.76484689003837,
 596.7269253645262,
 609.8084097831769,
 614.7216264441361,
 608.240361912658,
 603.3956346989169,
 592.0299911684662,
 549.8477010976336,
 525.5195285042266,
 434.69170736982505,
 340.40697150478536,
 272.08694555088937,
 208.61164681073478,
 112.55339473352205,
 21.343474577798588,
])

y16 = np.array([
159.29203539823015,
646.0176991150443,
674.7787610619469,
690.2654867256638,
712.3893805309734,
738.9380530973451,
763.2743362831858,
873.8938053097344,
942.4778761061947,
1196.9026548672564,
1329.646017699115,
1398.2300884955748,
1440.2654867256638,
1484.5132743362828,
1504.424778761062,
])


def plot2d_nointerp(df, tag):
    plt.clf()
    fig = plt.gcf()
    ax = plt.gca()
    plt.plot()

    excluded = df['exp'] < 1
    ax.plot(
            df['mphi'][excluded],
            df['mchi'][excluded],
            ls='none',
            color='b',
            marker='o'
            )
    ax.plot(
            df['mphi'][~excluded],
            df['mchi'][~excluded],
            ls='none',
            color='r',
            marker='o'
            )
    ax.plot(
            y16,
            x16,
            color='r',label='2016')

    outdir = f'./output/{tag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig.savefig(pjoin("tchan_2d_nointerp.pdf"))

def plot2d(df, tag):
    plt.clf()
    fig = plt.figure(figsize=(14,10))
    ax = plt.gca()
    plt.plot()

    excluded = df['exp'] < 1

    x = df['mphi']
    y = df['mchi']
    exp = df['exp']
    p1s = df['p1s']
    m1s = df['m1s']

    contours_filled = np.log10(np.logspace(-1,1,7))
    contours_line = [0]

    def get_x_y_z(x,y,z):
        ix, iy, iz = interpolate_rbf(x,y,z,maxval=2500  )
        iz [iy>ix] = 1e9 #* np.exp(-(iy/ix))
        if True: 
            iz = np.log10(iz)
            iz[iz<min(contours_filled)] = min(contours_filled)
        return ix, iy, iz
    ix, iy, iz = get_x_y_z(x, y, exp)
    CF = plt.contourf(ix, iy, iz, levels=contours_filled, cmap=cmap)
    cb = plt.colorbar()
    for c in CF.collections:
        c.set_edgecolor("face")

    args = dict(colors='black',linewidths=3,zorder=2,levels=contours_line,)
    cs = plt.contour(
                       ix, iy, iz, 
                       linestyles="solid",
                       **args)
    cs.collections[0].set_label('Median expected')
    cs2=plt.contour(
                *get_x_y_z(x,y,p1s),
                linestyles="--",
                **args)
    cs2.collections[0].set_label('Expected $\pm$ 1 s.d.')
    plt.contour(
                *get_x_y_z(x,y,m1s),
                linestyles="--",
                **args)
    cb.add_lines(cs)
    cb.set_label("95% CL expected limit on $\log_{10}(\mu)$")
    plt.clim([1e-1,1e1])

    # plt.plot(x,y, marker='+',color='k', ls='none')
    plt.plot(y16,x16,color='gold', label="2016 (36 fb$^{-1}$)", lw=3,zorder=1)


    ax.set_xlabel("$m_{\Phi}$ (GeV)")
    ax.set_ylabel("$m_{\chi}$ (GeV)")
    ax = plt.gca()
    ax.set_ylim(0,1200)

    plt.legend(loc='upper left', title= '\n'.join([
            f't-channel DM (S3D UR), $\lambda=1.0$'
        ]))
    hep.cms.label(data=True, year='2017-2018', lumi=101,paper=True)

    outdir = f'./output/{tag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig.savefig(pjoin(outdir, "tchan_2d.pdf"))


def main():
    tag = '2021_01_24_03Sep20v7_monojv_mistag_usepol1_testMCstat_default'
    df = pd.read_pickle(f"../input/{tag}/limit_df.pkl")
    df = df[(df.cl==0.95)& (~np.isnan(df.mphi)) &  (~np.isnan(df.mchi))]
    plot2d_nointerp(df, tag)
    plot2d(df, tag)

if __name__ == "__main__":
    main()