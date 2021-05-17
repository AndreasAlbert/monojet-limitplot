from tabulate import tabulate
import mplhep as hep
import os
from matplotlib import pyplot as plt
import uproot
from dataclasses import dataclass
plt.style.use(hep.style.CMS)

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

indir="input/2021-05-03_master"

channels = []
channels.append(Channel(
    file=os.path.join(indir, "higgsCombinemonojet_monov_nominal_combined.AsymptoticLimits.mH120.root"),
    dataset='1718',
    category='combined',
    xlabel='Combined'
))
channels.append(Channel(
    file=os.path.join(indir, "higgsCombine_monojet_combined.AsymptoticLimits.mH120.root"),
    dataset='1718',
    category='monojet',
    xlabel='Monojet'
))

channels.append(Channel(
    file=os.path.join(indir, "higgsCombinenominal_monovtight_combined.AsymptoticLimits.mH120.root"),
    dataset='1718',
    category='monovtightloose',
    xlabel='Mono-V\n(high-purity)'
))
channels.append(Channel(
    file=os.path.join(indir, "higgsCombinenominal_monovloose_combined.AsymptoticLimits.mH120.root"),
    dataset='1718',
    category='monovloose',
    xlabel='Mono-V\n(low-purity)'
))

fig = plt.gcf()
for i, channel in enumerate(channels):
    x = [i-0.5, i+0.5]

    plt.fill_between(
        x,
        2*[channel.p2s],
        2*[channel.m2s],
        color=colors['2s'],
        zorder=-5,
        label='Expected $\pm$ 2 s.d.' if i==0 else None
    )

    plt.fill_between(
        x,
        2*[channel.p1s],
        2*[channel.m1s],
        color=colors['1s'],
        zorder=-4,
        label='Expected $\pm$ 1 s.d.' if i==0 else None
    )

    plt.errorbar(
        i,
        channel.exp,
        yerr=0,
        xerr=0,
        marker='s',
        fillstyle='none',
        markersize=8,
        color=colors['exp'],
        label='95% CL expected' if i==0 else None
    )
    plt.errorbar(
        i,
        channel.obs,
        xerr=0.5,
        yerr=0,
        elinewidth=2,
        marker='o',
        color=colors['obs'],
        markersize=8,
        ls='-',
        label='95% CL observed' if i==0 else None
    )
    
plt.legend(loc='upper left')
hep.cms.label(data=True, year='2017,2018', lumi=101, paper=True)
plt.ylabel(r"BR(H$\rightarrow$ inv) = $\sigma_{obs}$ / $\sigma_{SM}(H)$")
plt.xticks(range(len(channels)), [channel.xlabel for channel in channels])

for ext in 'pdf','png':
    fig.savefig(f"hinv_categories.{ext}")


table = []
for c in reversed(channels):
    table.append((c.xlabel.replace("\n",""), c.exp, c.obs))
print(tabulate(table,floatfmt=".2f", headers=['Channel','Expected','Observed']))