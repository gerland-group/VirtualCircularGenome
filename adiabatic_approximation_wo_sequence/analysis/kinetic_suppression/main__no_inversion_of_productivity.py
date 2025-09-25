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
ratios_cvcgtot_cstocktot = np.logspace(-5, 1, 25)
Ls_vcg = [7,8,9]
Ls_vcg_sets = [(7,8),(7,9),(8,9)]

# compute data for single-length ensembles
dss_single_length = []
for L_vcg in Ls_vcg:
    ds = DataSet__SingleLength(L_vcg=L_vcg, c0_stock=c0_stock, \
                               ratios_cvcgtot_cstocktot=ratios_cvcgtot_cstocktot)
    ds.compute_effective_association_constants__up_to_duplex()
    ds.compute_effective_association_constants__reactive_triplex()
    ds.compute_all_data_points()
    ds.compute_total_inversion_concentration(f=0.5)
    ds.compute_asymptotic_fraction_consumed_oligos_by_monomer()
    dss_single_length.append(ds)

# plot
f, axs = plt.subplot_mosaic([['A','B','C']], figsize=(3*4.5,3.4), constrained_layout=True)

# set color for oligomer length
L2color = {7:'C0',8:'C1',9:'C2'}

# labels for panels
for letter in ['A','B','C']:
    axs[letter].text(-0.13, 1.15, letter, transform=axs[letter].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')

axs['A'].bar([1.0], [9.1e-5], color='grey')  
axs['A'].bar([2.0], [9.1e-6], color='grey') 
axs['A'].bar([7.0], [1e-6], color='grey') 
axs['A'].annotate('', xy=(7,2e-6), xytext=(7,1e-6), \
                  arrowprops=dict(facecolor='grey', shrink=0., edgecolor='grey'))
axs['A'].annotate('', xy=(7,5e-7), xytext=(7,1e-6), \
                  arrowprops=dict(facecolor='grey', shrink=0., edgecolor='white'))
axs['A'].set_yscale('log')    
axs['A'].set_xlim([0.2, 10.8])
axs['A'].set_ylim([1e-7, 1.3e-4]) 

axs['A'].set_xlabel(r'oligomer length $L$ (nt)')
axs['A'].set_ylabel(r'concentration $c(L)$ (M)')

for i, (L_vcg, ds) in enumerate(zip(Ls_vcg, dss_single_length)):
    axs['B'].plot(ratios_cvcgtot_cstocktot, ds.efficiencies, color=f'C{i}')
legends = {'$L_\mathrm{V}$ = 7 nt':Line2D(xdata=[], ydata=[], color='C0'), \
           '$L_\mathrm{V}$ = 8 nt':Line2D(xdata=[], ydata=[], color='C1'), \
           '$L_\mathrm{V}$ = 9 nt':Line2D(xdata=[], ydata=[], color='C2')}
axs['B'].set_xscale('log')
axs['B'].set_xlabel(r'concentration ratio $c_\mathrm{V}^\mathrm{tot}/c_\mathrm{F}^\mathrm{tot}$')
# axs['B'].set_ylabel('fraction of\nincorporated nucleotides')
axs['B'].set_ylabel('replication efficiency $\eta$')
axs['B'].legend(list(legends.values()), list(legends.keys()))


# fraction of oligomers extended by monomers for single-length VCGs
for i, (L_vcg, ds) in enumerate(zip(Ls_vcg, dss_single_length)):
    axs['C'].plot(ratios_cvcgtot_cstocktot, ds.fractions_consumed_oligos_by_monomer[L_vcg], \
                  label=r'$L_\mathrm{V}$ = %d nt' %L_vcg, color=f'C{i}')
    axs['C'].axvline(x=ds.c0_vcg_threshold/c0_stock, ymin=0, ymax=1, \
                     linestyle='dashed', color=f'C{i}')
    axs['C'].axhline(y=ds.fraction_consumed_oligos_by_monomer__asymptotic, xmin=0, xmax=1, \
                     color='grey', linestyle='dashed')
axs['C'].set_xscale('log')
axs['C'].set_xlabel(r'concentration ratio $c_\mathrm{V}^\mathrm{tot}/c_\mathrm{F}^\mathrm{tot}$')
axs['C'].set_ylabel('frac. of oligos in extension-\ncompetent state ' + r'$r_{1+\mathrm{V}}$')
axs['C'].legend()

plt.savefig('../../outputs/kinetic_suppression/main__no_inversion_productivity.pdf')
