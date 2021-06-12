import os
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from limitlib import fill_dummy_values, interpolate_rbf, dump_contour_to_txt

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


y16_exp = np.array([
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

x16_exp = np.array([
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
x16_obs = np.array([
156.36729047582043,
215.17500587268296,
284.0419700884842,
369.7230704982667,
447.004933051447,
542.7609427609447,
645.237908803802,
663.7512071620616,
685.7150314514672,
726.2234750606859,
746.4568162242597,
771.7484926787267,
810.5107926813366,
864.414167514944,
916.6288205047906,
982.3140970427803,
1041.2653668467628,
1090.1312870305126,
1140.6885391381522,
1194.623234933313,
1219.9201315480389,
1251.9719155378075,
1275.6114112703265,
1300.9135280453113,
1317.7955263226586,
1349.8290397515204,
1378.5059901338975,
1402.12721530551,
1420.7005455067472,
1424.104089995563,
1417.3883538224625,
1395.5237125779763,
1383.7574713543709,
1365.2807141179235,
1365.322475399995,
])

y16_obs = np.array([104.49612403100787,
159.06976744186045,
233.48837209302314,
317.8294573643411,
393.48837209302326,
490.23255813953494,
593.1782945736434,
595.6589147286821,
558.4496124031008,
508.8372093023256,
493.95348837209303,
475.34883720930236,
455.50387596899225,
440.62015503875966,
428.21705426356596,
414.5736434108527,
400.93023255813955,
379.84496124031,
355.0387596899225,
325.2713178294574,
304.18604651162786,
273.17829457364337,
239.68992248062023,
216.1240310077519,
193.79844961240315,
171.47286821705438,
144.1860465116281,
119.37984496124022,
93.33333333333326,
75.96899224806202,
67.28682170542652,
57.36434108527146,
48.68217054263573,
28.83720930232562,
8.992248062015506,
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
            x16_exp,
            y16_exp,
            color='r',label='2016 (Med. exp)', ls='--')
    ax.plot(
            x16_obs,
            y16_obs,
            color='r',label='2016 (Obs)')

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
    obs = df['obs']

    mask = ~((x==1600) & (y==650))
    x = x[mask]
    y = y[mask]
    exp = exp[mask]
    obs = obs[mask]
    p1s = p1s[mask]
    m1s = m1s[mask]

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
    ix_obs, iy_obs, iz_obs = get_x_y_z(x, y, obs)
    CF = plt.contourf(ix_obs, iy_obs, iz_obs, levels=contours_filled, cmap=cmap)
    cb = plt.colorbar()
    for c in CF.collections:
        c.set_edgecolor("face")

    args = dict(colors='black',linewidths=3,zorder=2,levels=contours_line,)
    cs_exp = plt.contour(
                       ix, iy, iz,
                       linestyles="--",
                       **args)
    cs_exp.collections[0].set_label('Median expected')

    cs_obs = plt.contour(
                       *get_x_y_z(x,y,obs),
                       linestyles="solid",
                       **args)
    cs_obs.collections[0].set_label('Observed')

    cs_p1s=plt.contour(
                *get_x_y_z(x,y,p1s),
                linestyles=":",
                **args)
    cs_p1s.collections[0].set_label(r'68% Expected')
    cs_m1s = plt.contour(
                *get_x_y_z(x,y,m1s),
                linestyles=":",
                **args)
    cb.add_lines(cs_obs)
    cb.set_label("95% CL observed upper limit on $\log_{10}(\mu)$")
    plt.clim([1e-1,1e1])

    # plt.plot(x,y, marker='+',color='k', ls='none')
    plt.plot(x16_obs,y16_obs,color='gold', label="2016 (36 fb$^{-1}$)", lw=3,zorder=1,ls='-')
    plt.plot(x16_exp,y16_exp,color='gold', lw=3,zorder=1,ls='--')


    ax.set_xlabel("$m_{\Phi}$ (GeV)")
    ax.set_ylabel("$m_{\chi}$ (GeV)")
    ax = plt.gca()
    ax.set_ylim(0,1200)

    plt.legend(loc='upper left')

    plt.text(2400,1100,'\n'.join([f'Fermion portal','(S3D U$_{{R}}$)', '$\lambda=1.0$']), ha='right',va='top')

    outdir = f'./output/{tag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hep.cms.label(data=True, year='2016-2018', lumi=137)
    for ext in 'pdf','png':
        fig.savefig(pjoin(outdir, f"tchan_2d.{ext}"))
    
    hep.cms.label(data=True, year='2016-2018', lumi=137, label="Preliminary")
    for ext in 'pdf','png':
        fig.savefig(pjoin(outdir, f"tchan_2d_preliminary.{ext}"))

    plt.plot(x,y,'+b')
    for ix, iy, iz in zip(x,y,obs):
        plt.text(ix, iy, f"{iz:.2f}",color='b',fontsize=10)

    for ext in 'pdf','png':
        fig.savefig(pjoin(outdir, f"tchan_2d_points.{ext}"))


    dump_contour_to_txt(cs_exp, pjoin(outdir, "contour_tchan_exp.txt"))
    dump_contour_to_txt(cs_obs, pjoin(outdir, "contour_tchan_obs.txt"))
    dump_contour_to_txt(cs_p1s, pjoin(outdir, "contour_tchan_p1s.txt"))
    dump_contour_to_txt(cs_m1s, pjoin(outdir, "contour_tchan_m1s.txt"))

def main():
    tag = '2021-05-03_master_default_default'
    df = pd.read_pickle(f"../input/{tag}/limit_df.pkl")
    df = df[(df.cl==0.95)& (~np.isnan(df.mphi)) &  (~np.isnan(df.mchi))]
    plot2d_nointerp(df, tag)
    plot2d(df, tag)

if __name__ == "__main__":
    main()