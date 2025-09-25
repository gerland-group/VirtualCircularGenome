#!/bin/env python3

import sys
sys.path.append('../../src/')
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
plt.style.use('seaborn-v0_8-colorblind')

from src__inversion_of_productivity__slim import DataSet__MultiLength

c0_mono_tot = 20e-3
ratios_cresttot_cmonotot = np.logspace(-5, 0, 20)
Ls_vcg = [6,8,10,12]
Ls_vcg_sets = [(6,8),(6,10),(6,12),(8,10),(8,12)]
print("constructing the complexes")
ds_multiple_lengths = DataSet__MultiLength(L_vcg_min=3, L_vcg_max=12, c0_mono_tot=c0_mono_tot, \
             ratios_cresttot_cmonotot=ratios_cresttot_cmonotot, \
             cmplxs_params=dict(l_unique=3, alphabet=4, Lmax_stock=2, comb_vcg=24, \
                                gamma_2m=-2.5, gamma_d=-1.25, include_triplexes=True, \
                                include_tetraplexes=True))
print("computing the concentrations")
ds_multiple_lengths.compute_all_data_points()

# plot
f, axs = plt.subplots(1,1,figsize=(4.5,3.4), constrained_layout=True)
for L_vcg in Ls_vcg:
    axs.plot(ratios_cresttot_cmonotot*c0_mono_tot, \
             ds_multiple_lengths.fractions_consumed_oligos_by_monomer[L_vcg], \
             label=r'$L$ = %d nt' %L_vcg)
axs.axvline(247e-6/c0_mono_tot, ymin=0, ymax=1, linestyle='dashed', color='grey')
axs.set_xscale('log')
axs.set_xlabel(r'concentration $\sum_{L=2}^{12} c^\mathrm{tot}(L)$ (M)')
axs.set_ylabel('frac. of oligos in extension-\ncompetent state ' + r'$r_{1+\mathrm{V}}\,(L)$')
axs.legend(loc='upper left', bbox_to_anchor=(0.53, 0.5))
plt.savefig('../../outputs/kinetic_suppression/SI__inversion_productivity__Szostak_parameters.pdf')