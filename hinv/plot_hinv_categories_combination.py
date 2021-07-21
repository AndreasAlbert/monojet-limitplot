from tabulate import tabulate
import mplhep as hep
import os
from matplotlib import pyplot as plt
import uproot
from dataclasses import dataclass
import matplotlib
plt.style.use(hep.style.CMS)
# matplotlib.rcParams['mathtext.scr']='Noto Sans'

@dataclass
class Channel:
    file: str
    dataset: str
    category: str
    xlabel: str
    exp: float = -1
    p1s: float = -1
    p2s: float = -1
    m1s: float = -1
    m2s: float = -1

    def __post_init__(self):
        f = uproot.open(self.file)
        tree = f['limit']
        vals = tree['limit'].array()

        self.p2s = vals[0]
        self.p1s = vals[1]
        self.exp = vals[2]
        self.m1s = vals[3]
        self.m2s = vals[4]
        self.obs = vals[5]
        

colors= {}
colors['1s'] = 'green'
colors['2s'] = 'orange'
colors['obs'] = 'k'
colors['exp'] = 'k'

indir="input/2021-05-03_master/combination"

channels = []
channels.append(Channel(
    file=os.path.join(indir, "higgsCombine_hinv_m125_CL0.95.AsymptoticLimits.mH120.root"),
    dataset='161718',
    category='combined',
    xlabel='Combined'
))
channels.append(Channel(
    file=os.path.join(indir, "higgsCombine_hinv_m125_combined_monov.AsymptoticLimits.mH120.root"),
    dataset='161718',
    category='monov',
    xlabel='Mono-V'
))
channels.append(Channel(
    file=os.path.join(indir, "higgsCombine_hinv_m125_combined_monojet.AsymptoticLimits.mH120.root"),
    dataset='161718',
    category='monojet',
    xlabel='Monojet'
))
# channels.append(Channel(
#     file=os.path.join(indir, "higgsCombine_monojet_combined.AsymptoticLimits.mH120.root"),
#     dataset='1718',
#     category='monojet',
#     xlabel='Monojet'
# ))

# channels.append(Channel(
#     file=os.path.join(indir, "higgsCombinenominal_monovtight_combined.AsymptoticLimits.mH120.root"),
#     dataset='1718',
#     category='monovtightloose',
#     xlabel='Mono-V\n(high-purity)'
# ))
# channels.append(Channel(
#     file=os.path.join(indir, "higgsCombinenominal_monovloose_combined.AsymptoticLimits.mH120.root"),
#     dataset='1718',
#     category='monovloose',
#     xlabel='Mono-V\n(low-purity)'
# ))

fig = plt.gcf()
for i, channel in enumerate(channels):
    x = [i-0.5, i+0.5]

    plt.fill_between(
        x,
        2*[channel.p2s],
        2*[channel.m2s],
        color=colors['2s'],
        zorder=-5,
        label=r'95% expected' if i==0 else None
    )

    plt.fill_between(
        x,
        2*[channel.p1s],
        2*[channel.m1s],
        color=colors['1s'],
        zorder=-4,
        label=r'68% expected' if i==0 else None
    )

    eb = plt.errorbar(
        i,
        channel.exp,
        xerr=0.5,
        marker='o',
        ls='none',
        fillstyle='none',
        markersize=8,
        color=colors['exp'],
        label='Median expected' if i==0 else None
    )
    eb[-1][0].set_linestyle('--')
    plt.errorbar(
        i,
        channel.obs,
        xerr=0.5,
        elinewidth=2,
        marker='o',
        color=colors['obs'],
        markersize=8,
        ls='-',
        label='Observed' if i==0 else None
    )
    
plt.legend(loc='upper left', title='95% CL upper limits')
plt.ylabel(r"$\mathcal{B}$ (H$\rightarrow$ inv) = $\sigma_{obs}$ / $\sigma_{SM}(H)$")
plt.xticks(range(len(channels)), [channel.xlabel for channel in channels])
plt.ylim(0,1)

plt.text(2.4,0.95,"CMS", fontweight='bold', ha='right',va='top',fontsize=30)
labels = hep.cms.label_base.exp_label(exp="",data=True, year='2016-2018', lumi=137)
for ext in 'pdf','png':
    fig.savefig(f"hinv_categories_combination.{ext}")

hep.cms.label(data=True, year='2016-2018', lumi=137, label="Preliminary")
for ext in 'pdf','png':
    fig.savefig(f"hinv_categories_combination_preliminary.{ext}")


table = []
for c in reversed(channels):
    table.append((c.xlabel.replace("\n",""), c.exp, c.obs))
print(tabulate(table,floatfmt=".3f", headers=['Channel','Expected','Observed']))