#!/bin/env python3

import sys
sys.path.append('../src/')
import pickle as pkl
import numpy as np
import os

import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')

from src__influence_of_LV import DatasetTheo, AnalyticPrediction, AnalyticPredictionNew

# global parameters/
c0_stock = 4 * 2.5e-5
include_simulation=True
include_analytics=True

# parameters panel B
Ls_vcg_short = [6,7,8]
ratios_cvcgtot_cstocktot = np.logspace(-5, -1, 25)
gamma_2m__B = -2.5

# parameters panel C, D, E, F
gammas_2m = [-1.25, -1.875, -2.5]
Ls_vcg = np.arange(5,16,1)

# dictionary to store the data theory
data_theo = {'varying_ratio_cvcgtot_cstocktot': \
                {L_vcg:
                    {'ratio_cvcgtot_cstocktot':[], \
                     'yield_ratio_cvflvs':[], \
                     'errorfree_ratio_cvflvs':[], \
                     'ratio__dimerization':[], \
                     'ratio__correct primer extension':[], \
                     'ratio__correct ligation':[], \
                     'ratio__incorrect ligation':[], \
                    }
                    for L_vcg in Ls_vcg_short
                }, \
             'varying_L_vcg': \
                {gamma_2m:
                    {'L_vcg':[], \
                     'errorfree_ratio_cvflvs_max':[], \
                     'ratio_cvcgequ_cstockequ':[], \
                     'ratio_cvcgtot_cstocktot':[], \
                     'ratio_cvcgequ_cstockequ_cto_lb':[], \
                     'ratio_cvcgequ_cstockequ_cto_ub':[], \
                     'ratio_cvcgtot_cstocktot_cto_lb':[], \
                     'ratio_cvcgtot_cstocktot_cto_ub':[], \
                    } 
                    for gamma_2m in gammas_2m}
            }

# compute the data via theoretical model
print("compute data for varying ratio_cvcgtot_cstocktot")
for L_vcg in Ls_vcg_short:
    print(f"L_vcg: {L_vcg}")
    for i in range(len(ratios_cvcgtot_cstocktot)):
        ds = DatasetTheo(L_vcg=L_vcg, gamma_2m=gamma_2m__B, c0_stock_all=c0_stock, \
                         c0_vcg_all=c0_stock*ratios_cvcgtot_cstocktot[i], \
                         optimize=False, compute_overlap=False)
        data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio_cvcgtot_cstocktot'].append(\
            ratios_cvcgtot_cstocktot[i])
        data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['errorfree_ratio_cvflvs'].append(\
            ds.cncs.errorfree_ratio_cvflvs_exact)
        data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['yield_ratio_cvflvs'].append(\
            ds.cncs.yield_ratio_cvflvs_exact)
        cs_nuc_cvflvs_tot = np.sum(np.asarray(list(ds.cncs.cs_nuc_cvflvs.values())))
        data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio__dimerization'].append(\
            ds.cncs.cs_nuc_cvflvs['ff_v_f_c']/cs_nuc_cvflvs_tot)
        data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio__correct primer extension'].append(\
            ds.cncs.cs_nuc_cvflvs['fv_v_v_c']/cs_nuc_cvflvs_tot)
        data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio__correct ligation'].append(\
            ds.cncs.cs_nuc_cvflvs['vv_v_v_c']/cs_nuc_cvflvs_tot)
        data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio__incorrect ligation'].append(\
            ds.cncs.cs_nuc_cvflvs['vv_v_v_f']/cs_nuc_cvflvs_tot)

print("compute data for varying L_vcg")
for gamma_2m in gammas_2m:
    print(f"gamma_2m: {gamma_2m}")
    for i in range(len(Ls_vcg)):

        ds4 = DatasetTheo(L_vcg=Ls_vcg[i], gamma_2m=gamma_2m, \
                                   c0_vcg_all=None, c0_stock_all=c0_stock, \
                                   optimize=True, compute_overlap=True)
        
        data_theo['varying_L_vcg'][gamma_2m]['L_vcg'].append(Ls_vcg[i])
        
        data_theo['varying_L_vcg'][gamma_2m]['errorfree_ratio_cvflvs_max'].append(\
            ds4.cncs.errorfree_ratio_cvflvs_exact_opt)
        
        data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgequ_cstockequ'].append(\
            ds4.cncs.ratio_cvcgequ_cstockequ_opt)
        data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgtot_cstocktot'].append(\
            ds4.cncs.ratio_cvcgtot_cstocktot_opt)
        data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgequ_cstockequ_cto_lb'].append(\
            ds4.cncs.ratio_cvcgequ_cstockequ_cto_lb)
        data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgequ_cstockequ_cto_ub'].append(\
            ds4.cncs.ratio_cvcgequ_cstockequ_cto_ub)
        data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgtot_cstocktot_cto_lb'].append(\
            ds4.cncs.ratio_cvcgtot_cstocktot_cto_lb)
        data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgtot_cstocktot_cto_ub'].append(\
            ds4.cncs.ratio_cvcgtot_cstocktot_cto_ub)

# read data obtained via the full kinetic simulation
f = open('../../inputs/fidelity_yield_kinetic_simulation.pkl', 'rb')
data_sim = pkl.load(f)
f.close()

# create analytic approximations
data_analytic = {}

# for gamma=-2.5
params25 = {'Ka0_FF_all': 11526.314025843336,
            'Lambda_FF_all': 2.4269904062796197,
            'Ka0_FV_all': 24.05953834857893,
            'Lambda_FV_all': 0.40000000000006625,
            'Ka0_FV_corr': 24.059538348425132,
            'Lambda_FV_corr': 0.40000000000000263,
            'Ka0_VV_all': 586.9032914985528,
            'Lambda_VV_all': 0.39481934737146684,
            'Ka0_VV_corr': 27.932863260916907,
            'Lambda_VV_corr': 0.39928733684592227,
            'Ka0_VV_err': 512.0,
            'Lambda_VV_err': 0.4}
ap = AnalyticPredictionNew(params=params25, Ls=Ls_vcg)
ap.compute_optimal_equilibrium_concentration_ratios()
ap.compute_characteristic_lengthscale()
ap.compute_maximal_efficiencies()
data_analytic[-2.5] = ap

# for gamma=-1.875
params18 = {'Ka0_FF_all': 3016.3281943349784,
            'Lambda_FF_all': 2.2167887466464022,
            'Ka0_FV_all': 36.65524318402315,
            'Lambda_FV_all': 0.5333333335269248,
            'Ka0_FV_corr': 36.655242806255664,
            'Lambda_FV_corr': 0.5333333333452178,
            'Ka0_VV_all': 586.5652854492072,
            'Lambda_VV_all': 0.5241529740063631,
            'Ka0_VV_corr': 27.924017081766234,
            'Lambda_VV_corr': 0.5320615196447731,
            'Ka0_VV_err': 511.99999999999994,
            'Lambda_VV_err': 0.5333333333333333}
ap = AnalyticPredictionNew(params=params18, Ls=Ls_vcg)
ap.compute_optimal_equilibrium_concentration_ratios()
ap.compute_characteristic_lengthscale()
ap.compute_maximal_efficiencies()
data_analytic[-1.875] = ap

# for gamma=-1.25
params12 = {'Ka0_FF_all': 752.1869965264069,
            'Lambda_FF_all': 1.9294771429620208,
            'Ka0_FV_all': 59.957087109154834,
            'Lambda_FV_all': 0.8000007133292345,
            'Ka0_FV_corr': 59.956067315108434,
            'Lambda_FV_corr': 0.800000042313373,
            'Ka0_VV_all': 585.2904712246299,
            'Lambda_VV_all': 0.7794375549789511,
            'Ka0_VV_corr': 27.89034744174379,
            'Lambda_VV_corr': 0.7970937673440974,
            'Ka0_VV_err': 511.99999999999994,
            'Lambda_VV_err': 0.8}
ap = AnalyticPredictionNew(params=params12, Ls=Ls_vcg)
ap.compute_optimal_equilibrium_concentration_ratios()
ap.compute_characteristic_lengthscale()
ap.compute_maximal_efficiencies()
data_analytic[-1.25] = ap

# read data for scaling with genome size
# this data can be produced by running the script 
# 'dataproduction__scaling_with_genome_length.py'
# in the same directory as this file
Lgs = np.loadtxt('../../outputs/changing_LV/data/Lg.txt')
Lss_cont = np.loadtxt('../../outputs/changing_LV/data/Ls_cont.txt')
Lss_disc = np.loadtxt('../../outputs/changing_LV/data/Ls_disc.txt')
Lcs_13 = np.loadtxt('../../outputs/changing_LV/data/Lc_gamma_1.25.txt')
Lcs_18 = np.loadtxt('../../outputs/changing_LV/data/Lc_gamma_1.88.txt')
Lcs_25 = np.loadtxt('../../outputs/changing_LV/data/Lc_gamma_2.50.txt')

#plot
xpos_letter = -0.13
ypos_letter = 1.15

f, axs = plt.subplots(3,3,figsize=(3*4.5, 3*3.4), constrained_layout=True)

axs[0,0].text(xpos_letter, ypos_letter, 'A', transform=axs[0,0].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')

axs[0,0].bar([1], [1e-4], color='grey')
axs[0,0].bar([7], [1e-6], color='grey')
axs[0,0].annotate('', xy=(7,2e-6), xytext=(7,1e-6), \
                  arrowprops=dict(facecolor='grey', shrink=0., edgecolor='grey'))
axs[0,0].annotate('', xy=(7,5e-7), xytext=(7,1e-6), \
                  arrowprops=dict(facecolor='grey', shrink=0., edgecolor='white'))
axs[0,0].set_yscale('log')
axs[0,0].set_ylim([1e-7, 1.3e-4])
axs[0,0].set_xlim([0.2, 10.8])
axs[0,0].set_xlabel(r'oligomer length $L$ (nt)')
axs[0,0].set_ylabel(r'concentration $c(L)$ (M)')

axs[0,1].text(xpos_letter, ypos_letter, 'B', transform=axs[0,1].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')
for i, L_vcg in enumerate(Ls_vcg_short):
    axs[0,1].plot(data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio_cvcgtot_cstocktot'], \
                  data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['yield_ratio_cvflvs'], \
                  label=r"$L_\mathrm{V}$ = %d nt" %L_vcg, color=f"C{i}")
    if include_simulation:
        if L_vcg in data_sim['varying_ratio_cvcgtot_cstocktot'].keys():
            axs[0,1].errorbar(data_sim['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio_cvcgtot_cstocktot'], \
                              data_sim['varying_ratio_cvcgtot_cstocktot'][L_vcg]['yield_ratio_avg'], \
                              data_sim['varying_ratio_cvcgtot_cstocktot'][L_vcg]['yield_ratio_std'], \
                              color=f"C{i}", linestyle='', marker='o', capsize=5)
axs[0,1].set_xscale('log')
axs[0,1].set_xlabel(r'concentration ratio $c^\mathrm{tot}_\mathrm{V}/c^\mathrm{tot}_\mathrm{F}$')
axs[0,1].set_ylabel('yield $y$')
axs[0,1].legend()

axs[0,2].text(xpos_letter, ypos_letter, 'C', transform=axs[0,2].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')
sum = np.zeros(len(ratios_cvcgtot_cstocktot))
type2label = {'dimerization':'(F+F)', 'correct primer extension':'(F+V), c', 'correct ligation':'(V+V), c', 'incorrect ligation':'(V+V), f'}
typethermo2typesim = {'dimerization':'oligomerization_correct', \
                      'correct primer extension':'primer_extension_correct', \
                      'correct ligation':'templated_ligation_correct', \
                      'incorrect ligation':'templated_ligation_false'}
for i, type in enumerate(\
    [el for el in data_theo['varying_ratio_cvcgtot_cstocktot'][6].keys() if '__' in el]):
    axs[0,2].plot(data_theo['varying_ratio_cvcgtot_cstocktot'][6]['ratio_cvcgtot_cstocktot'], \
                  np.asarray(data_theo['varying_ratio_cvcgtot_cstocktot'][6][type]), \
                  label=type2label[type.split('__')[1]], color=f"C{i}")
    if include_simulation:
        typesim = typethermo2typesim[type.split('__')[1]]
        axs[0,2].errorbar(data_sim['varying_ratio_cvcgtot_cstocktot'][6]['ratio_cvcgtot_cstocktot'], \
                            data_sim['varying_ratio_cvcgtot_cstocktot'][6][f"{typesim}_avg"], \
                            data_sim['varying_ratio_cvcgtot_cstocktot'][6][f"{typesim}_std"], \
                            color=f"C{i}", linestyle='', marker='o', capsize=5)
axs[0,2].set_xscale('log')
axs[0,2].set_xlabel(r'concentration ratio $c^\mathrm{tot}_\mathrm{V}/c^\mathrm{tot}_\mathrm{F}$')
axs[0,2].set_ylabel('ligation share $s$')
axs[0,2].legend()

axs[1,0].text(xpos_letter, ypos_letter, 'D', transform=axs[1,0].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')
for i, L_vcg in enumerate(Ls_vcg_short):
    axs[1,0].plot(data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio_cvcgtot_cstocktot'], \
                  np.asarray(data_theo['varying_ratio_cvcgtot_cstocktot'][L_vcg]['errorfree_ratio_cvflvs']), \
                  label=r"$L_\mathrm{V}$ = %d nt" %L_vcg, color=f"C{i}")
    if include_simulation:
        if L_vcg in data_sim['varying_ratio_cvcgtot_cstocktot'].keys():
            axs[1,0].errorbar(data_sim['varying_ratio_cvcgtot_cstocktot'][L_vcg]['ratio_cvcgtot_cstocktot'], \
                              np.asarray(data_sim['varying_ratio_cvcgtot_cstocktot'][L_vcg]['errorfree_ratio_avg']), \
                              data_sim['varying_ratio_cvcgtot_cstocktot'][L_vcg]['errorfree_ratio_std'], \
                              color=f"C{i}", linestyle='', marker='o', capsize=5)
axs[1,0].set_xscale('log')
axs[1,0].set_xlabel(r'concentration ratio $c^\mathrm{tot}_\mathrm{V}/c^\mathrm{tot}_\mathrm{F}$')
axs[1,0].set_ylabel('replication efficiency $\eta$')
axs[1,0].legend()

axs[1,1].text(xpos_letter, ypos_letter, 'E', transform=axs[1,1].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')

axs[2,0].text(xpos_letter, ypos_letter, 'F', transform=axs[2,0].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')
for i, gamma_2m in enumerate(gammas_2m):
    axs[2,0].plot(data_theo['varying_L_vcg'][gamma_2m]['L_vcg'], \
                  data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgequ_cstockequ'], \
                  color=f"C{i}", label=r"$\gamma$ = %1.2f $k_\mathrm{B} T$" %gamma_2m)
    axs[2,0].fill_between(data_theo['varying_L_vcg'][gamma_2m]['L_vcg'], \
                  data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgequ_cstockequ_cto_lb'], \
                  data_theo['varying_L_vcg'][gamma_2m]['ratio_cvcgequ_cstockequ_cto_ub'], \
                  alpha=0.3, color=f"C{i}")
    if include_analytics and gamma_2m in data_analytic:
        axs[2,0].plot(data_analytic[gamma_2m].Ls, data_analytic[gamma_2m].rs_opt, \
                      color=f"grey", linestyle='dashed')
axs[2,0].set_yscale('log')
axs[2,0].set_xlabel(r'length of VCG oligomers $L_\mathrm{V}$ (nt)')
axs[2,0].set_ylabel('conc. ratio '+ r'$(c^\mathrm{eq}_\mathrm{V}/c^\mathrm{eq}_\mathrm{F})_\mathrm{opt}$')
axs[2,0].legend()

axs[2,1].text(xpos_letter, ypos_letter, 'G', transform=axs[2,1].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')
for i, gamma_2m in enumerate(gammas_2m):
    axs[2,1].plot(data_theo['varying_L_vcg'][gamma_2m]['L_vcg'], \
                  data_theo['varying_L_vcg'][gamma_2m]['errorfree_ratio_cvflvs_max'], \
                  label=r"$\gamma$ = %1.2f $k_\mathrm{B} T$" %gamma_2m, color=f"C{i}")
    if include_analytics and gamma_2m in data_analytic:
        axs[2,1].axvline(x=data_analytic[gamma_2m].L_c, ymin=0, ymax=1, \
                         linestyle='dotted', color=f"C{i}")
        axs[2,1].plot(data_analytic[gamma_2m].Ls_etas, data_analytic[gamma_2m].etas, \
                      linestyle='dashed', color='grey')
axs[2,1].set_xlabel(r'length of VCG oligomers $L_\mathrm{V}$ (nt)')
axs[2,1].set_ylabel(r'replication efficiency $\eta_\mathrm{max}$')
axs[2,1].legend()


axs[2,2].text(xpos_letter, ypos_letter, 'H', transform=axs[2,2].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')

axs[2,2].plot(Lgs, Lss_disc, label=r"$L_{\rm U}$")
axs[2,2].plot(Lgs, Lcs_25, label=r'$L_\mathrm{V}$ at $\eta_\mathrm{max} = 95$%')
axs[2,2].set_xlabel(r'genome length $L_{\rm G}$ (nt)')
axs[2,2].set_ylabel('length (nt)')
axs[2,2].set_xscale('log')
axs[2,2].legend()

f.savefig(f'../../outputs/changing_LV/main__influence_LV__cfeedall_{c0_stock:1.2e}.pdf')
plt.show()
