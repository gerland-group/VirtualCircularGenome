#!/bin/env python3

import sys
sys.path.append('../../src/')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.style.use('seaborn-v0_8-colorblind')
mpl.rcParams['font.size'] = 12.0

from multiprocessing import Pool

from ComplexConstructor import *
from ConcentrationComputer import *

class DataSet:

    def __init__(self, L_vcg, L_vcg_min, L_vcg_max, L_stock_max, \
                 c0_vcg_all, c0_stock_all, Lambda_vcg, Lambda_stock, \
                 optimize, include_tetraplex):

        self.cmplxs = ComplexConstructor(l_unique=3, alphabet=4, \
                        L_vcg=L_vcg, L_vcg_min=L_vcg_min, L_vcg_max=L_vcg_max, \
                        Lmax_stock=L_stock_max, comb_vcg=32, \
                        gamma_2m=-2.5, gamma_d=-1.25, \
                        include_tetraplexes=include_tetraplex)
        
        if not optimize:
            self.cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                            c0_vcg_all_oligos=c0_vcg_all, \
                            Lambda_vcg=Lambda_vcg, \
                            c0_stock_all_oligos=c0_stock_all, \
                            Lambda_stock=Lambda_stock)
                                       
            # compute equilibrium concentrations          
            self.cncs.compute_equilibrium_concentration_log()
            self.cncs.compute_concentrations_added_nucleotides_productive_cvflvs()
            self.cncs.compute_error_ratio_cvflvs_exact()
            self.cncs.compute_error_ratio_cvflvs_fv_v_v_f_only_exact()
            self.cncs.compute_errorfree_ratio_cvflvs_exact()
            self.cncs.compute_errorfree_ratio_cvflvs_ff_v_v_c_missing_exact()
            self.cncs.compute_errorfree_ratio_cvflvs_ff_v_v_c_only_exact()
        
        if optimize:
            self.cncs = ConcentrationOptimizerAddedNucsExact(\
                cmplxs=self.cmplxs, \
                Lambda_vcg=Lambda_vcg, \
                c0_stock_all_oligos=c0_stock_all, \
                Lambda_stock=Lambda_stock)
            self.cncs.compute_optimal_total_concentration_ratio_cvflvs_exact()

class AsymptoticEfficiency:

    def __init__(self, params, kappa_feed):
        
        self.Ka0_L1Lc = params['Ka0_L1Lc']
        self.Ka0_L1Lf = params['Ka0_L1Lf']
        self.Ka0_L2Lc = params['Ka0_L2Lc']
        self.Ka0_L2Lf = params['Ka0_L2Lf']
        self.kappa_feed = kappa_feed
    
    def compute_asymptotic_efficiency(self):
        num = self.Ka0_L1Lc + 2*self.Ka0_L2Lc*np.exp(-self.kappa_feed)
        denom = self.Ka0_L1Lc + 2*(self.Ka0_L2Lc+self.Ka0_L2Lf)*np.exp(-self.kappa_feed)
        self.eta_max = num/denom

# changeable parameters
c0_stock = 1e-4

ratios_c0vcg_c0stock_theo_1 = np.logspace(-6, 0, 150)
Ls_stock_max = [1,2]

Ls_vcg_long = np.arange(3, 15, 1)

Ls_vcg_short = np.arange(5, 11, 1)

kappas_stock_short = np.linspace(0, np.log(1000), 4)

Ls_vcg_min = [3]
Ls_vcg_max = np.arange(5, 11, 1)
Ls_vcg_sets = []
for L_vcg_min in Ls_vcg_min:
    for L_vcg_max in Ls_vcg_max:
        if L_vcg_max >= L_vcg_min:
            Ls_vcg_sets.append((L_vcg_min, L_vcg_max))

data = {
    'varying_Lstockmax__yield_ratios':{
        L_stock_max:{
            'ratios_c0vcg_c0stock':[], \
            'errorfree_ratio_cvflvs_exact_3':[], \
            'error_ratio_cvflvs_fv_v_v_f_only_exact_3':[], \
            'errorfree_ratio_cvflvs_exact_4':[], \
            'error_ratio_cvflvs_fv_v_v_f_only_exact_4':[] }
        for L_stock_max in Ls_stock_max}, \
    'varying_kappastock__Lvcgmin_eq_Lvcgmax':{
        kappa_stock:{
            'L_vcg':[], \
            'errorfree_ratio_cvflvs_exact_3':[], \
            'errorfree_ratio_cvflvs_exact_4':[] }
        for kappa_stock in kappas_stock_short}, \
    'varying_kappastock__asymptotic':{
        kappa_stock:None \
        for kappa_stock in kappas_stock_short}
    }


L_vcg = 8
for L_stock_max in Ls_stock_max:
    print(f"L_stock_max: {L_stock_max}")
    for i in range(len(ratios_c0vcg_c0stock_theo_1)):

        ds3 = DataSet(L_vcg=L_vcg, L_vcg_min=L_vcg, L_vcg_max=L_vcg, \
                      L_stock_max=L_stock_max, \
                      c0_vcg_all=c0_stock*ratios_c0vcg_c0stock_theo_1[i], \
                      c0_stock_all=c0_stock, \
                      Lambda_vcg=np.inf, Lambda_stock=1/np.log(10), \
                      optimize=False, include_tetraplex=False)
        
        ds4 = DataSet(L_vcg=L_vcg, L_vcg_min=L_vcg, L_vcg_max=L_vcg, \
                      L_stock_max=L_stock_max, \
                      c0_vcg_all=c0_stock*ratios_c0vcg_c0stock_theo_1[i], \
                      c0_stock_all=c0_stock, \
                      Lambda_vcg=np.inf, Lambda_stock=1/np.log(10), \
                      optimize=False, include_tetraplex=True)
        
        data['varying_Lstockmax__yield_ratios'][L_stock_max]['ratios_c0vcg_c0stock'].append(\
            ratios_c0vcg_c0stock_theo_1[i])
        
        data['varying_Lstockmax__yield_ratios'][L_stock_max]['error_ratio_cvflvs_fv_v_v_f_only_exact_3'].append(\
            ds3.cncs.error_ratio_cvflvs_fv_v_v_f_only_exact)
        data['varying_Lstockmax__yield_ratios'][L_stock_max]['errorfree_ratio_cvflvs_exact_3'].append(\
            ds3.cncs.errorfree_ratio_cvflvs_exact)
        data['varying_Lstockmax__yield_ratios'][L_stock_max]['error_ratio_cvflvs_fv_v_v_f_only_exact_4'].append(\
            ds4.cncs.error_ratio_cvflvs_fv_v_v_f_only_exact)
        data['varying_Lstockmax__yield_ratios'][L_stock_max]['errorfree_ratio_cvflvs_exact_4'].append(\
            ds4.cncs.errorfree_ratio_cvflvs_exact)

for kappa_stock in kappas_stock_short:
    for L_vcg in Ls_vcg_short:
        print(f"kappa_stock: {kappa_stock}, L_vcg: {L_vcg}")
        ds3 = DataSet(L_vcg=L_vcg, L_vcg_min=L_vcg, L_vcg_max=L_vcg, L_stock_max=2, \
                     c0_vcg_all=None, c0_stock_all=c0_stock, \
                     Lambda_stock=1/kappa_stock, Lambda_vcg=np.inf, \
                     optimize=True, include_tetraplex=False)
        
        ds4 = DataSet(L_vcg=L_vcg, L_vcg_min=L_vcg, L_vcg_max=L_vcg, L_stock_max=2, \
                     c0_vcg_all=None, c0_stock_all=c0_stock, \
                     Lambda_stock=1/kappa_stock, Lambda_vcg=np.inf, \
                     optimize=True, include_tetraplex=True)
        
        data['varying_kappastock__Lvcgmin_eq_Lvcgmax'][kappa_stock]['L_vcg'].append(L_vcg)
        
        data['varying_kappastock__Lvcgmin_eq_Lvcgmax'][kappa_stock]['errorfree_ratio_cvflvs_exact_3'].append(\
            ds3.cncs.errorfree_ratio_cvflvs_exact_opt)
        data['varying_kappastock__Lvcgmin_eq_Lvcgmax'][kappa_stock]['errorfree_ratio_cvflvs_exact_4'].append(\
            ds4.cncs.errorfree_ratio_cvflvs_exact_opt)

for kappa_stock in kappas_stock_short:
    # the parameters used as input are determined using the script
    # SI__scaling_effective_association_constants.py
    ea = AsymptoticEfficiency(params={'Ka0_L1Lc': 24.05953834850841,
                                      'Ka0_L1Lf': 172705.36762057745,
                                      'Ka0_L2Lc': 22.01488458707896,
                                      'Ka0_L2Lf': 48.00000000566951}, \
                              kappa_feed=kappa_stock)
    ea.compute_asymptotic_efficiency()
    data['varying_kappastock__asymptotic'][kappa_stock] = ea.eta_max

# plot
f, axs = plt.subplots(2,3,figsize=(3*4.5,2*3.2), constrained_layout=True)

axs[0,0].text(-0.15, 1.15, 'A', transform=axs[0,0].transAxes, \
              fontsize=18, fontweight='bold', va='top', ha='right')
axs[0,1].text(-0.18, 1.15, 'B', transform=axs[0,1].transAxes, \
              fontsize=18, fontweight='bold', va='top', ha='right')
axs[0,2].text(-0.10, 1.15, 'C', transform=axs[0,2].transAxes, \
              fontsize=18, fontweight='bold', va='top', ha='right')
axs[1,0].text(-0.15, 1.15, 'D', transform=axs[1,0].transAxes, \
              fontsize=18, fontweight='bold', va='top',ha='right')

axs[0,0].bar([1,2], [1e-4,1e-5], color='grey')
axs[0,0].plot(np.array([1.6,2.6]), 1.2*np.array([1e-4, 1e-5]), color='grey', \
              linestyle='dashed')
axs[0,0].text(x=2.4, y=4e-5, s=r'$\sim \kappa_\mathrm{F}^{-1}$', color='grey')
axs[0,0].bar([7], 1e-6, color='grey')
axs[0,0].annotate('', xy=(7,2e-6), xytext=(7,1e-6), \
                  arrowprops=dict(facecolor='grey', shrink=0., edgecolor='grey'))
axs[0,0].annotate('', xy=(7,5e-7), xytext=(7,1e-6), \
                  arrowprops=dict(facecolor='grey', shrink=0., edgecolor='white'))
axs[0,0].set_yscale('log')
axs[0,0].set_ylim([1e-7, 1.7e-4])
axs[0,0].set_xlim([0.2, 10.8])
axs[0,0].set_xlabel(r'oligomer length $L$ (nt)')
axs[0,0].set_ylabel(r'concentration $c(L)$ (M)')

linestyles = ['solid', 'dashed']
for i, L_stock_max in enumerate(Ls_stock_max):
    axs[0,1].plot(np.asarray(data['varying_Lstockmax__yield_ratios'][L_stock_max]['ratios_c0vcg_c0stock']), \
                    data['varying_Lstockmax__yield_ratios'][L_stock_max]['errorfree_ratio_cvflvs_exact_4'], \
                    color='C0', linestyle=linestyles[i])
    axs[0,1].plot(np.asarray(data['varying_Lstockmax__yield_ratios'][L_stock_max]['ratios_c0vcg_c0stock']), \
                    data['varying_Lstockmax__yield_ratios'][L_stock_max]['error_ratio_cvflvs_fv_v_v_f_only_exact_4'], \
                    color='C1', linestyle=linestyles[i])
for i in range(len(linestyles)):
    axs[0,1].plot([], label='$L^\mathrm{max}_\mathrm{F}$ = %d nt' %(i+1), color='grey', \
                  linestyle=linestyles[i])

axs[0,1].set_xscale('log')
axs[0,1].set_xlabel(r'concentration ratio $c^\mathrm{tot}_\mathrm{V}/c^\mathrm{tot}_\mathrm{F}$')
y_label_parts = ['replication efficiency $\eta$', 'ligation share $s$[(F+V), f]']
colors = ['C0', 'C1']

# Define the position for the custom y-label
x_pos = -0.11  # x position, slightly to the left of the axis
y_pos = 0.5   # y position, 1.0 is at the top of the y-axis

# Add each part of the y-label with different colors
for i, (text, color) in enumerate(zip(y_label_parts, colors)):
    axs[0,1].text(x_pos-i*0.06, y_pos, text, transform=axs[0,1].transAxes, rotation=90, 
            color=color, ha='right', va='center', fontsize=12)
axs[0,1].set_ylabel('')

axs[0,1].set_ylim([-0.03,1.01])
axs[0,1].legend(loc='upper left')

for i, kappa_stock in enumerate(kappas_stock_short):
    axs[0,2].plot(data['varying_kappastock__Lvcgmin_eq_Lvcgmax'][kappa_stock]['L_vcg'], \
                  data['varying_kappastock__Lvcgmin_eq_Lvcgmax'][kappa_stock]['errorfree_ratio_cvflvs_exact_4'], \
                  label=r"$\kappa_\mathrm{F}$ = %1.1f" %kappa_stock, \
                  color=f"C{i}")
    axs[0,2].plot(data['varying_kappastock__Lvcgmin_eq_Lvcgmax'][kappa_stock]['L_vcg'], \
                  data['varying_kappastock__asymptotic'][kappa_stock] \
                  * np.ones(len(data['varying_kappastock__Lvcgmin_eq_Lvcgmax'][kappa_stock]['L_vcg'])), \
                  color=f"C{i}", linestyle='dashed')
axs[0,2].set_xlabel('length of VCG oligomers $L_\mathrm{V}$ (nt)')
axs[0,2].set_ylabel('replication efficiency $\eta_\mathrm{max}$')
axs[0,2].legend(loc='upper right', bbox_to_anchor=(0.975, 0.55))

plt.savefig(f'../../outputs/changing_LF/main__influence_LF__cfeedall_{c0_stock:1.2e}.pdf')
