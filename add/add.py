from tabulate import tabulate
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
    f = interp1d(x,y, fill_value="extrapolate", kind="linear")

    minfun = lambda tmp : (f(tmp)-value)**2

    result = minimize(minfun, x0=7)

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
            res = find_intersection(x, np.log10(y), 0)
            mdlimits[d][quantile] = max(res.x)

        fig = plt.gcf()
        fig.clf()
        hep.cms.label(data=True, year='2016-2018', lumi=137, paper=True)
        ax = fig.gca()

        xinterp = np.linspace(min(x),max(x),50)
        ax.plot(x, np.log10(idf.exp),zorder=2,marker='o',ls="--",fillstyle='none',color='k', markersize=10, label="Median expected",lw=2)
        ax.plot(x, np.log10(idf.obs),zorder=2,marker='o',color='k', markersize=10, label="Observed",lw=2)
        ax.fill_between(x, np.log10(idf.m1s), np.log10(idf.p1s),color='green',zorder=1, label=r'68% expected')
        ax.fill_between(x, np.log10(idf.m2s), np.log10(idf.p2s),color='orange',zorder=0, label=r'95% expected')

        ax.plot(2*[mdlimits[d]['obs']],[-0.1,0.1],ls='--',color='r')
        ax.plot(2*[mdlimits[d]['exp']],[-0.1,0.1],ls='--',color='r')
        ax.plot(2*[mdlimits[d]['p1s']],[-0.1,0.1],ls='--',color='r')
        ax.plot(2*[mdlimits[d]['p2s']],[-0.1,0.1],ls='--',color='r')
        ax.plot(2*[mdlimits[d]['m1s']],[-0.1,0.1],ls='--',color='r')
        ax.plot(2*[mdlimits[d]['m2s']],[-0.1,0.1],ls='--',color='r')

        ymin = min(idf.exp)
        ymax = max(idf.exp)

        # plt.plot([mdlimits[d]['exp'],mdlimits[d]['exp']]2,[ymin,ymax],color='red',lw=2)
        plt.plot([min(x), max(x)],[0, 0],color='red',lw=2)

        ax = plt.gca()
        ax.set_xlabel("$M_{D}$ (TeV)")
        ax.set_ylabel("95% CL upper limit on $log_{10}(\mu)$")
        ax.set_ylim(-1,1)
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
    hep.cms.label(data=True, year='2016-2018', lumi=137, paper=True)
    eb = plt.errorbar(x, exp, xerr=0.5, yerr=0, zorder=2,marker='o',fillstyle='none',color='k',ls="None", label="Median expected", markersize=10, linewidth=2)
    eb[-1][0].set_linestyle('--')
    plt.errorbar(x, obs, xerr=0.5, yerr=0, zorder=2,marker='o',color='k',ls="None", label="Observed", markersize=10, linewidth=2)
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

    table = []
    for md,o,e in zip(x, obs, exp):
        table.append((md, o, e))
    print(tabulate(table, headers=['d','MD obs.','MD exp.'],floatfmt='.1f'))

def calculate_d7(df):
    df=df.append(
    df[df.d==6].assign(d=7), ignore_index=True
    )

    df.loc[df.d==7,'tag'] = df[df.d==7]['tag'].apply(lambda x: x.replace("d6","d7"))


    factors = {
                5: 0.6561275062616915,
                6: 0.5749991701751788,
                7: 0.49690580278979174,
                8: 0.45637598491539816,
                9: 0.4111812551421096
                }

    for md in 5, 6, 7:
        mask = (df.d==7) & (df.md==md)
        for key in 'exp','obs','p1s','p2s','m1s','m2s':
            df.loc[mask,key] = df[mask][key] / factors[md]
    return df

def main():
    inpath = "../input/2021-05-03_master_default_default/limit_df.pkl"
    df  = pd.read_pickle(inpath)
    df = df[df.cl==0.95]
    df = df[~np.isnan(df.d)]

    df = calculate_d7(df)
    tag = inpath.split("/")[-2]
    add_d_limits(df, tag)
    add_md_limits(tag)

if __name__ == "__main__":
    main()