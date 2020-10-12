import os
import mplhep as hep
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from collections import defaultdict
pjoin = os.path.join
plt.style.use(hep.style.CMS)
def find_intersection(x, y, value):
    f = interp1d(x,y, fill_value="extrapolate")

    minfun = lambda tmp : (f(tmp)-value)**2

    result = minimize(minfun, x0=np.mean(x))

    return result


brazilgreen = "green"
brazilyellow = "orange"


def add_d_limits(df, tag):
    outdir = f'./output/{tag}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    mdlimits = defaultdict(dict)
    for d in map(int,set(df.d)):
        idf = df[df.d==d]

        x = idf['md']
        for quantile in ['exp','obs','p1s','m1s','p2s','m2s']:
            y = idf[quantile]
            res = find_intersection(x, y, 1)
            mdlimits[d][quantile] = res.x[0]

        fig = plt.gcf()
        fig.clf()
        hep.cms.cmslabel(data=True, year='2016-2018', lumi=137)
        ax = fig.gca()
        ax.plot(x, idf.exp,zorder=2,marker='o',fillstyle='none',color='k', markersize=10, label="Median expected",lw=2)
        ax.fill_between(x, idf.m1s, idf.p1s,color='green',zorder=1, label=r'68% expected')
        ax.fill_between(x, idf.m2s, idf.p2s,color='orange',zorder=0, label=r'95% expected')

        ymin = min(idf.exp)
        ymax = max(idf.exp)

        # plt.plot([mdlimits[d]['exp'],mdlimits[d]['exp']]2,[ymin,ymax],color='red',lw=2)
        plt.plot([min(x), max(x)],[1,1],color='red',lw=2)

        ax = plt.gca()
        ax.set_xlabel("$M_{D}$ (TeV)")
        ax.set_ylabel("95% CL upper limit on the signal strength $\mu$")
        ax.set_ylim(0,3)
        plt.legend(title=f'ADD, d = {d:.0f}', loc='upper left')
        for ext in ['pdf','png']:
            plt.gcf().savefig(pjoin(outdir, f"d{d}.{ext}"))

    with open(pjoin(outdir,"mdlimits.pkl"),"wb") as f:
        pickle.dump(dict(mdlimits), f)

def binned_fill(x, low, high, **kwargs):
    label = kwargs.pop("label")

    first = True
    width = 0.49
    for ix, il, ih in zip(x,low, high):
        plt.fill_between([ix-width,ix+width],[il,il],[ih,ih],label=label if first else None, **kwargs)
        first = False

def limits_2016():
    x = [
    2,
    3,
    4,
    5,
    6]
    obs = [
    10.011816838995568,
    7.494830132939439,
    6.360413589364846,
    5.757754800590842,
    5.261447562776958,
    ]
    exp=[
    10.54357459379616,
    7.79615952732644,
    6.555391432791728,
    6.005908419497784,
    5.385524372230428]

    return x, exp, obs

def add_md_limits(tag):
    outdir = f'./output/{tag}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(pjoin(outdir,"mdlimits.pkl"),"rb") as f:
        mdlimits = pickle.load(f)

    x = sorted(mdlimits.keys())

    obs = [mdlimits[ix]['obs'] for ix in x]
    exp = [mdlimits[ix]['exp'] for ix in x]
    p1s = [mdlimits[ix]['p1s'] for ix in x]
    p2s = [mdlimits[ix]['p2s'] for ix in x]
    m1s = [mdlimits[ix]['m1s'] for ix in x]
    m2s = [mdlimits[ix]['m2s'] for ix in x]

    plt.gcf().clf()
    hep.cms.cmslabel(data=True, year='2016-2018', lumi=137)
    eb = plt.errorbar(x, exp, xerr=0.5, yerr=0, zorder=2,marker='o',fillstyle='none',color='k',ls="None", label="Median expected", markersize=10, linewidth=2)
    eb[-1][0].set_linestyle('--')
    binned_fill(x, m1s, p1s,zorder=1,color=brazilgreen, label=r'68% expected')
    binned_fill(x, m2s, p2s,zorder=0,color=brazilyellow, label=r'95% expected')


    ax = plt.gca()
    ax.set_xlabel("Number of extra dimensions")
    ax.set_ylabel("95% CL lower limit on $M_{D}$ (TeV)")
    ax.set_ylim(0,20)
    plt.legend()
    for ext in ['pdf','png']:
        plt.gcf().savefig(pjoin(outdir, f"md.{ext}"))

    x16, exp16, obs16 = limits_2016()

    plt.errorbar(x16, obs16, xerr=0.5, yerr=0, marker='o',color='blue', label="2016 observed", linewidth=2, markersize=10,ls="none")
    plt.legend()
    for ext in ['pdf','png']:
        plt.gcf().savefig(pjoin(outdir, f"md_with2016.{ext}"))

def main():
    inpath = "../dmsimp/input/2020-10-04_03Sep20v7/limit_df.pkl"
    df  = pd.read_pickle(inpath)
    df = df[df.cl==0.95]
    df = df[~np.isnan(df.d)]
    tag = inpath.split("/")[-2]
    add_d_limits(df, tag)
    add_md_limits(tag)

if __name__ == "__main__":
    main()