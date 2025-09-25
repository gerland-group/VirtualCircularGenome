#!/bin/env python3

import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
import matplotlib.pyplot as plt
# plt.style.use('seaborn-colorblind')
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

        self.eff_all_max = np.max(self.eff_all)

params_strong = [
    (3,4,'none',5), \
    (3,4,'none',6), \
    (3,4,'none',7), \
    (3,4,'none',8), \
    (3,4,'none',9), \
    (3,4,'none',10), \
    (3,4,'none',11), \
    (3,4,'none',12), \

    (2,4,'strongly_nonuniform',5), \
    (2,4,'strongly_nonuniform',6), \
    (2,4,'strongly_nonuniform',7), \
    (2,4,'strongly_nonuniform',8), \
    
    (3,6,'strongly_nonuniform',6), \
    (3,6,'strongly_nonuniform',7), \
    (3,6,'strongly_nonuniform',8), \
    (3,6,'strongly_nonuniform',9), \
    
    (2,6,'strongly_nonuniform',6), \
    (2,6,'strongly_nonuniform',7), \
    (2,6,'strongly_nonuniform',8), \
    (2,6,'strongly_nonuniform',9), \
    
    
    (3,8,'strongly_nonuniform',8), \
    (3,8,'strongly_nonuniform',9), \
    (3,8,'strongly_nonuniform',10), \

    (2,8,'strongly_nonuniform',8), \
    (2,8,'strongly_nonuniform',9), \
    (2,8,'strongly_nonuniform',10), \

    (3,10,'strongly_nonuniform',10), \
    (3,10,'strongly_nonuniform',11), \
    (3,10,'strongly_nonuniform',12), \

    (2,10,'strongly_nonuniform',10), \
    (2,10,'strongly_nonuniform',11), \
    (2,10,'strongly_nonuniform',12)
]

params_weak = [
    (3,4,'none',5), \
    (3,4,'none',6), \
    (3,4,'none',7), \
    (3,4,'none',8), \
    (3,4,'none',9), \
    (3,4,'none',10), \
    (3,4,'none',11), \
    (3,4,'none',12), \

    (2,4,'weakly_nonuniform',5), \
    (2,4,'weakly_nonuniform',6), \
    (2,4,'weakly_nonuniform',7), \
    (2,4,'weakly_nonuniform',8), \
    
    (3,6,'weakly_nonuniform',6), \
    (3,6,'weakly_nonuniform',7), \
    (3,6,'weakly_nonuniform',8), \
    (3,6,'weakly_nonuniform',9), \
    
    (2,6,'weakly_nonuniform',6), \
    (2,6,'weakly_nonuniform',7), \
    (2,6,'weakly_nonuniform',8), \
    (2,6,'weakly_nonuniform',9), \
    
    (3,8,'weakly_nonuniform',8), \
    (3,8,'weakly_nonuniform',9), \
    (3,8,'weakly_nonuniform',10), \

    (2,8,'weakly_nonuniform',8), \
    (2,8,'weakly_nonuniform',9), \
    (2,8,'weakly_nonuniform',10), \

    (3,10,'weakly_nonuniform',10), \
    (3,10,'weakly_nonuniform',11), \
    (3,10,'weakly_nonuniform',12), \

    (2,10,'weakly_nonuniform',10), \
    (2,10,'weakly_nonuniform',11), \
    (2,10,'weakly_nonuniform',12)
]

dss_strong = {param:DataSet(l_gen=64, L1=param[0], L2=param[1], bias=param[2], \
                            l_oligo_F=1, l_oligo_V=param[3]) \
              for param in params_strong}
dss_weak = {param:DataSet(l_gen=64, L1=param[0], L2=param[1], bias=param[2], \
                          l_oligo_F=1, l_oligo_V=param[3]) for param in params_weak}

grouped_params_strong = [
    (3,4,'none'), \
    (2,4,'strongly_nonuniform'), \
    (3,6,'strongly_nonuniform'), \
    (2,6,'strongly_nonuniform'), \
    (3,8,'strongly_nonuniform'), \
    (2,8,'strongly_nonuniform'), \
    (3,10,'strongly_nonuniform'), \
    (2,10,'strongly_nonuniform')
]

grouped_params_weak = [
    (3,4,'none'), \
    (2,4,'weakly_nonuniform'), \
    (3,6,'weakly_nonuniform'), \
    (2,6,'weakly_nonuniform'), \
    (3,8,'weakly_nonuniform'), \
    (2,8,'weakly_nonuniform'), \
    (3,10,'weakly_nonuniform'), \
    (2,10,'weakly_nonuniform')
]


fig, axs = plt.subplots(1,2,figsize=(2*4.5,3.2), constrained_layout=True)

axs[0].text(-0.13, 1.15, 'A', transform=axs[0].transAxes,
    fontsize=18, fontweight='bold', va='top', ha='right')
axs[1].text(-0.13, 1.15, 'B', transform=axs[1].transAxes, \
    fontsize=18, fontweight='bold', va='top', ha='right')

for grouped_param in grouped_params_strong:
    l_oligos = [param[3] for param in dss_strong.keys() if param[0] == grouped_param[0]\
                and param[1] == grouped_param[1]]
    effs = [dss_strong[param].eff_all_max for param in dss_strong.keys() if param[0] == grouped_param[0] \
            and param[1] == grouped_param[1]]
    axs[0].plot(l_oligos, effs, color=f'C{(grouped_param[1]-4)//2}', \
                linestyle='solid' if grouped_param[0] == 3 else 'dotted')
axs[0].set_xlabel(r'length of VCG oligomers $L_\mathrm{V}$ (nt)')
axs[0].set_ylabel(r'replication efficiency $\eta_\mathrm{max}$')
handles = [mpl.lines.Line2D([], [], color='C0'), \
           mpl.lines.Line2D([], [], color='C1'), \
           mpl.lines.Line2D([], [], color='C2'), \
           mpl.lines.Line2D([], [], color='C3'), \
           mpl.lines.Line2D([], [], color='grey', linestyle='dotted'), \
           mpl.lines.Line2D([], [], color='grey', linestyle='solid')]
labels = [r'$L_{\rm U}\,=\,$4$\,$nt', r'$L_{\rm U}\,=\,$6$\,$nt', r'$L_{\rm U}\,=\,$8$\,$nt', \
          r'$L_{\rm U}\,=\,$10$\,$nt', r'$L_{\rm E}\,=\,$2$\,$nt', r'$L_{\rm E}\,=\,$3$\,$nt']
axs[0].legend(handles, labels)

for grouped_param in grouped_params_weak:
    l_oligos = [param[3] for param in dss_weak.keys() if param[0] == grouped_param[0]\
                and param[1] == grouped_param[1]]
    effs = [dss_weak[param].eff_all_max for param in dss_weak.keys() if param[0] == grouped_param[0] \
            and param[1] == grouped_param[1]]
    axs[1].plot(l_oligos, effs, color=f'C{(grouped_param[1]-4)//2}', \
                linestyle='solid' if grouped_param[0] == 3 else 'dotted')
axs[1].set_xlabel(r'length of VCG oligomers $L_\mathrm{V}$ (nt)')
axs[1].set_ylabel(r'replication efficiency $\eta_\mathrm{max}$')
handles = [mpl.lines.Line2D([], [], color='C0'), \
           mpl.lines.Line2D([], [], color='C1'), \
           mpl.lines.Line2D([], [], color='C2'), \
           mpl.lines.Line2D([], [], color='C3'), \
           mpl.lines.Line2D([], [], color='grey', linestyle='dotted'), \
           mpl.lines.Line2D([], [], color='grey', linestyle='solid')]
labels = [r'$L_{\rm U}\,=\,$4$\,$nt', r'$L_{\rm U}\,=\,$6$\,$nt', r'$L_{\rm U}\,=\,$8$\,$nt', \
          r'$L_{\rm U}\,=\,$10$\,$nt', r'$L_{\rm E}\,=\,$2$\,$nt', r'$L_{\rm E}\,=\,$3$\,$nt']
axs[1].legend(handles, labels)


plt.savefig('../../outputs/plots/SI__eff_v_LV.pdf')

plt.show()

