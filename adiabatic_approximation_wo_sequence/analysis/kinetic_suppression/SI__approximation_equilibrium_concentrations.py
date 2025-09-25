#!/bin/env python3

import sys
sys.path.append('../../src/')
sys.path.append('./')
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
plt.style.use('seaborn-v0_8-colorblind')

from src__inversion_of_productivity import DataSet__MultiLength

ratios_cvcgtot_cstocktot = np.logspace(-5, 0, 25)
ds = DataSet__MultiLength(L_vcg_min=3, L_vcg_max=9, c0_stock=1e-4, \
             ratios_cvcgtot_cstocktot=ratios_cvcgtot_cstocktot, \
             Ls_vcg_sets=[(6,8),(6,9),(7,8),(7,9),(8,9)], \
             number_iterations=10)
ds.compute_effective_association_constants__up_to_duplex()
ds.compute_effective_association_constants__reactive_triplex()
ds.compute_alphas_and_betas_for_approximate_equilibration()
ds.compute_all_data_points()
ds.compute_total_inversion_concentration__all_pairs()

f, axs = plt.subplots(1,2,figsize=(2*4.5,3.2), constrained_layout=True)

axs[0].text(-0.20, 1.15, 'A', transform=axs[0].transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')
axs[0].plot(np.asarray(list(ds.rmses.keys())), np.asarray(list(ds.rmses.values())))
axs[0].set_xlabel('number of iterations')
axs[0].set_ylabel('root of relative\nmean squared error')

axs[1].text(-0.17, 1.15, 'B', transform=axs[1].transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')
for i, L_vcg in enumerate([6,7,8,9]):
    axs[1].plot(ratios_cvcgtot_cstocktot, ds.css_ss_comb_equ[L_vcg]/ds.css_ss_comb_tot[L_vcg], \
                color=f'C{i}', label=r'$L$ = %d' %L_vcg)
    axs[1].plot(ratios_cvcgtot_cstocktot, ds.css_ss_comb_equ__approx0[L_vcg]/ds.css_ss_comb_tot[L_vcg], \
                color='grey', linestyle='dotted')
    axs[1].plot(ratios_cvcgtot_cstocktot, ds.css_ss_comb_equ__approx1[L_vcg]/ds.css_ss_comb_tot[L_vcg], \
                linestyle='dashed', color=f'grey')
legend_handles = [plt.Line2D(xdata=[], ydata=[], color=f'C{i}') for i in range(4)]
legend_handles.append(plt.Line2D(xdata=[], ydata=[], color='grey', linestyle='solid'))
legend_handles.append(plt.Line2D(xdata=[], ydata=[], color='grey', linestyle='dotted'))
legend_handles.append(plt.Line2D(xdata=[], ydata=[], color='grey', linestyle='dashed'))
axs[1].legend(legend_handles, ["$L$ = 6", "$L$ = 7", "$L$ = 8", "$L$ = 9", \
                               "exact", "0th iter.", "1st iter."])
axs[1].set_ylim([4e-2, 1.2e0])
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel('concentration ratio $c^\mathrm{tot}_\mathrm{V}/c^\mathrm{tot}_\mathrm{F}$')
axs[1].set_ylabel('conc. ratio $c^\mathrm{eq}(L)/c^\mathrm{tot}(L)$')
plt.savefig('../../outputs/kinetic_suppression/' \
            + 'SI__inversion_productivity__approximation_equilibrium_concentrations.pdf')
plt.show()
