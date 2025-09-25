#!/bin/env python3

import sys
sys.path.append('../../src/')
sys.path.append('./')
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
plt.style.use('seaborn-v0_8-colorblind')

from src__inversion_of_productivity import DataSet__SingleLength, DataSet__MultiLength

def plot_multicolor(ax, x, y, colors):
    w = 4
    line1, = ax.plot(x, y, color=colors[0], linestyle=(0, (w, w, w, w)))
    line2, = ax.plot(x, y, linestyle=(0, (0, w, w, 2*w)), color=colors[1])
    return (line1, line2)

c0_stock = 1e-4
ratios_cvcgtot_cstocktot = np.logspace(-5, 0, 25)
Ls_vcg = [7,8,9]
Ls_vcg_sets = [(7,8),(7,9),(8,9)]

# compute data for multi-length ensemble
L_vcg_min, L_vcg_max = 3, 9
ds_multiple_lengths = DataSet__MultiLength(L_vcg_min=L_vcg_min, L_vcg_max=L_vcg_max, \
                                           c0_stock=c0_stock, \
                                           ratios_cvcgtot_cstocktot=ratios_cvcgtot_cstocktot, \
                                           number_iterations=2, \
                                           Ls_vcg_sets=Ls_vcg_sets)
ds_multiple_lengths.compute_effective_association_constants__up_to_duplex()
ds_multiple_lengths.compute_effective_association_constants__reactive_triplex()
ds_multiple_lengths.compute_alphas_and_betas_for_approximate_equilibration()
ds_multiple_lengths.compute_all_data_points()
ds_multiple_lengths.compute_total_inversion_concentration__all_pairs()

# plot
f, axs = plt.subplot_mosaic([['A','B','C'],\
                             ['D','E','F']], figsize=(3*4.5,2*3.4), constrained_layout=True)

# set color for oligomer length
L2color = {7:'C0',8:'C1',9:'C2',10:'C3'}

# labels for panels
for letter in ['A', 'B', 'D', 'E', 'F']:
    axs[letter].text(-0.13, 1.15, letter, transform=axs[letter].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')
axs['C'].text(-0.17, 1.15, 'C', transform=axs['C'].transAxes, \
              fontsize=18, fontweight='bold', va='top', ha='right')

# concentration profile
axs['A'].bar([1.0], [9.1e-5], color='red')
axs['A'].bar([2.0], [9.1e-6], color='grey')
for i in range(3, 7): 
    axs['A'].bar([i], [5e-7], color='grey')
for i in range(0,3):  
    axs['A'].bar([7+i], [5e-7], color=f'C{i}')
axs['A'].set_yscale('log')
axs['A'].set_ylim([1e-7, 1.3e-4])
axs['A'].set_xlim([0.2, 9.8])
axs['A'].set_xlabel(r'oligomer length $L$ (nt)')
axs['A'].set_ylabel(r'concentration $c(L)$ (M)')

# fraction of oligomer extended by monomers for mixed-length VCGs
for L_vcg in Ls_vcg:
    axs['B'].plot(ratios_cvcgtot_cstocktot, ds_multiple_lengths.fractions_consumed_oligos_by_monomer[L_vcg], \
                  label=r'$L_\mathrm{V}$ = %d nt' %L_vcg)
for L_vcg_set in ds_multiple_lengths.Ls_vcg_sets:
    axs['B'].axvline(x=ds_multiple_lengths.c0_vcg_thresholds[L_vcg_set]/c0_stock, ymin=0, ymax=1, \
                     color=L2color[L_vcg_set[0]], linestyle=(0, (5,8)))
    axs['B'].axvline(x=ds_multiple_lengths.c0_vcg_thresholds[L_vcg_set]/c0_stock, ymin=0, ymax=1, \
                     color=L2color[L_vcg_set[1]], linestyle=(5, (5,8)))
axs['B'].set_xscale('log')
axs['B'].set_xlabel(r'concentration ratio $c_\mathrm{V}^\mathrm{tot}/c_\mathrm{F}^\mathrm{tot}$')
#axs['B'].set_ylabel('fraction of monomer-\nextended oligomers ' + r'$r_{1+\mathrm{V}}\,(L_\mathrm{V})$')
axs['B'].set_ylabel('frac. of oligos in extension-\ncompetent state ' + r'$r_{1+\mathrm{V}}\,(L_\mathrm{V})$')
axs['B'].legend()

lines_all = {}
for L in [8,9]:
    ctot8 = ds_multiple_lengths.css_ss_comb_tot[8]
    ctot9 = ds_multiple_lengths.css_ss_comb_tot[9]
    lines = plot_multicolor(axs['C'], ratios_cvcgtot_cstocktot, \
                    (ds_multiple_lengths.css_cvfolp[f'({L}, 1, 8)_c'] + ds_multiple_lengths.css_cvfolp[f'({L}, 1, 8)_f'])/ctot8, \
                    colors=[L2color[L], L2color[8]])
    lines_all[r'$\frac{1|8}{%d}$' %L] = lines
    lines = plot_multicolor(axs['C'], ratios_cvcgtot_cstocktot, \
                    (ds_multiple_lengths.css_cvfolp[f'({L}, 1, 9)_c'] + ds_multiple_lengths.css_cvfolp[f'({L}, 1, 9)_f'])/ctot9, \
                    colors=[L2color[L], L2color[9]])
    lines_all[r'$\frac{1|9}{%d}$' %L] = lines
axs['C'].set_xscale('log')
axs['C'].set_xlabel(r'concentration ratio $c_\mathrm{V}^\mathrm{tot}/c_\mathrm{F}^\mathrm{tot}$')
axs['C'].set_ylabel('frac. of oligos in extension-\ncompetent state ' + r'$r_{1+\mathrm{V}}\,\left( \frac{1\,|\,L_\mathrm{E}}{L_\mathrm{T}} \right)$')
axs['C'].legend(list(lines_all.values()), list(lines_all.keys()))

for L_vcg in Ls_vcg:
    axs['D'].plot(ratios_cvcgtot_cstocktot, ds_multiple_lengths.css_ss_comb_equ[L_vcg]/ds_multiple_lengths.css_ss_comb_tot[L_vcg], \
                  label='$L_\mathrm{V}$ = %d nt' %L_vcg)
axs['D'].set_xscale('log')
axs['D'].set_yscale('log')
axs['D'].set_xlabel(r'concentration ratio $c_\mathrm{V}^\mathrm{tot}/c_\mathrm{F}^\mathrm{tot}$')
axs['D'].set_ylabel(r'conc. ratio $\left.c^\mathrm{eq}(L)\right/c(L)$')
axs['D'].legend()

plt.savefig('../../outputs/kinetic_suppression/main__inversion_productivity.pdf')
