#!/bin/env python3

import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')
import numpy as np
import os
import re

class DataSet:
    def __init__(self, l_gen, L1, L2, bias, l_oligo_F, l_oligo_V):
        self.l_gen = l_gen
        self.L1 = L1
        self.L2 = L2
        self.bias = bias
        self.l_oligo_F = l_oligo_F
        self.l_oligo_V = l_oligo_V
        self.dirpath = f'../../outputs/data/Lgen_{l_gen}__L1_{L1}__L2_{L2}__{bias}/ls_{l_oligo_F}_{l_oligo_V}/'
        self.filepaths = sorted([self.dirpath+el for el in os.listdir(self.dirpath) if re.match(r'concentrations\_productive', el)], \
                                 key=lambda el: float(el.split('_')[-1].split('.txt')[0]))
        self.read_productive_concentration__all_data_points()
        self.compute_efficiency()

    def read_productive_concentration__single_data_point(self, filepath):
        with open(filepath, 'r') as f:
            filestring = f.read()
        lines = [el for el in filestring.split('\n') if el != '']
        lines = lines[1:]
        cs_prod = np.zeros(5)
        for i, line in enumerate(lines):
            value = line.split('\t\t')[1]
            cs_prod[i] = float(value)
        return cs_prod

    def read_productive_concentration__all_data_points(self):
        self.cs_prod = np.zeros((len(self.filepaths), 5))
        for i, filepath in enumerate(self.filepaths):
            self.cs_prod[i] = self.read_productive_concentration__single_data_point(filepath)

    def compute_efficiency(self):
        csFtot = np.asarray([float(path.split('_')[-2]) for path in self.filepaths])
        csVtot = np.asarray([float(path.split('_')[-1].split('.txt')[0]) for path in self.filepaths])
        eff_all__numerator = self.l_oligo_F*self.cs_prod[:,2] + self.l_oligo_V*self.cs_prod[:,4]
        eff_all__denominator = self.l_oligo_F*self.cs_prod[:,0] + self.l_oligo_F*self.cs_prod[:,1] + self.l_oligo_F*self.cs_prod[:,2] \
                               + self.l_oligo_V*self.cs_prod[:,3] + self.l_oligo_V*self.cs_prod[:,4]
        eff_all = eff_all__numerator/eff_all__denominator
        eff_Fonly__numerator = self.l_oligo_F*self.cs_prod[:,2]
        eff_Fonly__denominator = self.l_oligo_F*self.cs_prod[:,0] + self.l_oligo_F*self.cs_prod[:,1] + self.l_oligo_F*self.cs_prod[:,2]
        eff_Fonly = eff_Fonly__numerator/eff_Fonly__denominator

        self.csFtot = csFtot
        self.csVtot = csVtot
        self.eff_all = eff_all
        self.eff_Fonly = eff_Fonly

params = [
    (3,4,'none',5), \
    (3,4,'none',6), \
    (3,4,'none',7), \
    (3,4,'none',8), \
    (3,4,'none',9), \
    (3,4,'none',10), \
    (3,4,'none',11), \
    (3,4,'none',12), \
    (2,6,'strongly_nonuniform',6), \
    (2,6,'strongly_nonuniform',7), \
    (2,6,'strongly_nonuniform',8), \
    (2,8,'strongly_nonuniform',8), \
    (2,8,'strongly_nonuniform',9), \
    (2,8,'strongly_nonuniform',10), \
    (2,10,'strongly_nonuniform',10), \
    (2,10,'strongly_nonuniform',11), \
    (2,10,'strongly_nonuniform',12)
]

dss = {param:DataSet(l_gen=64, L1=param[0], L2=param[1], bias=param[2], l_oligo_F=1, l_oligo_V=param[3]) \
       for param in params}


fig, axs = plt.subplots(1,3, figsize=(3*4.5,3.2), constrained_layout=True)
# labels for panels
axs[0].text(-0.13, 1.15, 'A', transform=axs[0].transAxes,
    fontsize=18, fontweight='bold', va='top', ha='right')
axs[1].text(-0.13, 1.15, 'B', transform=axs[1].transAxes, \
    fontsize=18, fontweight='bold', va='top', ha='right')
axs[2].text(-0.13, 1.15, 'C', transform=axs[2].transAxes, \
    fontsize=18, fontweight='bold', va='top', ha='right')

for param in dss.keys():
    if param[1] == 4 and param[3] >= 6 and param[3] <= 8:
        axs[0].plot(dss[param].csVtot/dss[param].csFtot, dss[param].eff_all, \
                color=f'C{param[3]-6}', label=r'$L_\mathrm{V}\,=\,$%d$\,$nt' %param[3])
    if param[1] == 6 and param[3] >= 6 and param[3] <= 8:
        axs[0].plot(dss[param].csVtot/dss[param].csFtot, dss[param].eff_all, color=f'C{param[3]-6}', linestyle='dotted')
#legend1 = axs[0].legend(loc=1, handlelength=1.)
handles, labels = axs[0].get_legend_handles_labels()
handles = [mpl.lines.Line2D([], [], linestyle='solid', color='grey'), \
           mpl.lines.Line2D([], [], linestyle='dotted', color='grey')] + handles
labels = [r'$L_{\rm U}\,=\,$4$\,$nt', r'$L_{\rm U}\,=\,$6$\,$nt'] + labels
legend2 = axs[0].legend(handles, labels, loc=1, handlelength=1.)
# axs[0].add_artist(legend1)
axs[0].set_xscale('log')
axs[0].set_xlabel(r'concentration ratio $c^\mathrm{tot}_\mathrm{V} / c^\mathrm{tot}_\mathrm{F}$')
axs[0].set_ylabel(r'replication efficiency $\eta$')

for param in dss.keys():
    if param[1] == 4 and param[3] >= 8 and param[3] <= 10:
        axs[1].plot(dss[param].csVtot/dss[param].csFtot, dss[param].eff_all, \
            color=f'C{param[3]-8}', label=r'$L_\mathrm{V}\,=\,$%d$\,$nt' %param[3])
    if param[1] == 8 and param[3] >= 8 and param[3] <= 10:
        axs[1].plot(dss[param].csVtot/dss[param].csFtot, dss[param].eff_all, \
            color=f'C{param[3]-8}', linestyle='dotted')
handles, labels = axs[1].get_legend_handles_labels()
handles = [mpl.lines.Line2D([], [], linestyle='solid', color='grey'), \
           mpl.lines.Line2D([], [], linestyle='dotted', color='grey')] + handles
labels = [r'$L_{\rm U}\,=\,$4$\,$nt', r'$L_{\rm U}\,=\,$8$\,$nt'] + labels
axs[1].legend(handles, labels, handlelength=1., loc='lower center', bbox_to_anchor=(0.45,0))
axs[1].set_xscale('log')
axs[1].set_xlabel(r'concentration ratio $c^\mathrm{tot}_\mathrm{V} / c^\mathrm{tot}_\mathrm{F}$')
axs[1].set_ylabel(r'replication efficiency $\eta$')

for param in dss.keys():
    if param[1] == 4 and param[3] >= 10 and param[3] <= 12:
        axs[2].plot(dss[param].csVtot/dss[param].csFtot, dss[param].eff_all, \
                      color=f'C{param[3]-10}', label=r'$L_\mathrm{V}\,=\,$%d$\,$nt' %param[3])
    if param[1] == 10 and param[3] >= 10 and param[3] <= 12:
        axs[2].plot(dss[param].csVtot/dss[param].csFtot, dss[param].eff_all, \
                      color=f'C{param[3]-10}', linestyle='dotted')
handles, labels = axs[2].get_legend_handles_labels()
handles = [mpl.lines.Line2D([], [], linestyle='solid', color='grey'), \
           mpl.lines.Line2D([], [], linestyle='dotted', color='grey')] + handles
labels = [r'$L_{\rm U}\,=\,$4$\,$nt', r'$L_{\rm U}\,=\,$10$\,$nt'] + labels
axs[2].legend(handles, labels, handlelength=1., loc=3)
axs[2].set_xscale('log')
axs[2].set_xlabel(r'concentration ratio $c^\mathrm{tot}_\mathrm{V} / c^\mathrm{tot}_\mathrm{F}$')
axs[2].set_ylabel(r'replication efficiency $\eta$')

plt.savefig('../../outputs/plots/SI__efficiency_vs_cVtot__strongly_nonuniform__L1_2__summaryplot.pdf', \
            bbox_inches='tight')
plt.show()

