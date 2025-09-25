#!/bin/env python3

import sys
sys.path.append('../../src/')
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 12.0
plt.style.use('seaborn-v0_8-colorblind')
from scipy.optimize import curve_fit

from ComplexConstructor import *
from ConcentrationComputer import *
from src__scaling_effective_association_constants import f_lin, f_exp, DataSet

dss = []

for gamma in [-2.5, -1.875, -1.25]:
    ds = DataSet(gamma=gamma)
    ds.construct_effective_association_constants()   
    ds.perform_curve_fit()
    # ds.print_fitted_parameters()
    ds.check_validity_of_approximation()
    ds.create_dictionary_of_fitted_parameters()
    dss.append(ds)

# plot the scalings
f, axs = plt.subplots(1,3,figsize=(3*4.5, 3.2), constrained_layout=True)

for i, letter in enumerate(['A', 'B', 'C']):
    axs[i].text(-0.13, 1.15, letter, transform=axs[i].transAxes, \
                fontsize=18, fontweight='bold', va='top', ha='right')

for i, ds in enumerate(dss):
    axs[0].scatter(ds.Ls_vcg, ds.Ka_FF_all, marker='o', \
                   edgecolor=f'C{i}', facecolor='white', \
                   label=rf'$\gamma$ = {ds.gamma:1.2f}' + r' $k_\mathrm{B} T$')
    axs[0].plot(ds.Ls_vcg_cont, f_lin(ds.Ls_vcg_cont, ds.Ka0_FF_all, ds.Ka0_FF_all/ds.Lambda_FF_all), \
                  color=f'C{i}')
    axs[0].set_xlabel(r'oligomer length $L_\mathrm{V}$')
    axs[0].set_ylabel(r'$\mathcal{K}^a_\mathrm{F+F}~(1/\mathrm{M}^3)$')
    axs[0].legend()
    
    axs[1].scatter(ds.Ls_vcg, ds.Ka_VV_err, marker='o', \
                     edgecolor=f'C{i}', facecolor='white', \
                     label=rf'$\gamma$ = {ds.gamma:1.2f}' + r' $k_\mathrm{B} T$')
    axs[1].plot(ds.Ls_vcg_cont, f_exp(ds.Ls_vcg_cont, ds.Ka0_VV_err, ds.Lambda_VV_err), \
                  color=f'C{i}')
    axs[1].set_xlabel(r'oligomer length $L_\mathrm{V}$')
    axs[1].set_ylabel(r'$\mathcal{K}^a_{\mathrm{V+V}, f}~(1/\mathrm{M}^3)$')
    axs[1].set_yscale('log')
    axs[1].legend()
    
    axs[2].scatter(ds.Ls_vcg, ds.Ka_FV_all, marker='o', \
                     edgecolor=f'C{i}', facecolor='white', \
                     label=rf'$\gamma$ = {ds.gamma:1.2f}' + r' $k_\mathrm{B} T$')
    axs[2].plot(ds.Ls_vcg_cont, f_exp(ds.Ls_vcg_cont, ds.Ka0_FV_all, ds.Lambda_FV_all), \
                  color=f'C{i}')
    axs[2].set_xlabel(r'oligomer length $L_\mathrm{V}$')
    axs[2].set_ylabel(r'$\mathcal{K}^a_\mathrm{F+V}~(1/\mathrm{M}^3)$')
    axs[2].set_yscale('log')
    axs[2].legend()

plt.savefig('../../outputs/changing_LV/SI__scaling_effective_association_constants_comparison.pdf')
