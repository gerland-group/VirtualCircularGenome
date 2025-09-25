#!/bin/env python3

import sys
sys.path.append('../../src/')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')
from scipy.optimize import curve_fit

from ComplexConstructor import *
from ConcentrationComputer import *


class EffectiveAssociationConstants:

    def __init__(self, Ls_vcg):
        
        self.Ls_vcg = Ls_vcg
        self.Ls_vcg_cont = np.linspace(self.Ls_vcg[0], self.Ls_vcg[-1], 100)
        self.Kas_L1Lc = np.zeros(len(self.Ls_vcg))
        self.Kas_L1Lf = np.zeros(len(self.Ls_vcg))
        self.Kas_L2Lc = np.zeros(len(self.Ls_vcg))
        self.Kas_L2Lf = np.zeros(len(self.Ls_vcg))
        
    def compute_effective_association_constants__single_Lvcg(self, L_vcg):

        cmplxs = ComplexConstructor(l_unique=3, alphabet=4, \
                        L_vcg=L_vcg, L_vcg_min=L_vcg, L_vcg_max=L_vcg, \
                        Lmax_stock=2, comb_vcg=32, gamma_2m=-2.5, gamma_d=-1.25)
        cmplxs.identify_reaction_types_cvfolp_all_complexes()
        
        cncs = ConcentrationComputer(cmplxs=cmplxs, \
                            c0_vcg_all_oligos=None, \
                            Lambda_vcg=np.inf, \
                            c0_stock_all_oligos=None, \
                            Lambda_stock=np.inf)
        # cncs.compute_ratio_concentration_strand_to_reference_strand()
        cncs.set_ratio_concentration_strand_to_reference_strand_to_one()
        cncs.compute_complex_weights()
        cncs.compute_weights_productive_cvfolp()

        Ka_L1Lc = cncs.ws_cvfolp[f'({L_vcg}, 1, {L_vcg})_c']
        Ka_L1Lf = cncs.ws_cvfolp[f'({L_vcg}, 1, {L_vcg})_f']
        Ka_L2Lc = cncs.ws_cvfolp[f'({L_vcg}, 2, {L_vcg})_c']
        Ka_L2Lf = cncs.ws_cvfolp[f'({L_vcg}, 2, {L_vcg})_f']

        return Ka_L1Lc, Ka_L1Lf, Ka_L2Lc, Ka_L2Lf
    
    def compute_effective_association_constants__all_Lsvcg(self):

        for i, L_vcg in enumerate(self.Ls_vcg):
            self.Kas_L1Lc[i], self.Kas_L1Lf[i], self.Kas_L2Lc[i], self.Kas_L2Lf[i] = \
                self.compute_effective_association_constants__single_Lvcg(L_vcg)

    def compute_replication_efficiencies(self, kappa_F):
        num = self.Kas_L1Lc + 2/4*self.Kas_L2Lc*np.exp(-kappa_F)
        denom = self.Kas_L1Lc + self.Kas_L1Lf + 2/4*(self.Kas_L2Lc + self.Kas_L2Lf)*np.exp(-kappa_F)
        self.etas = num/denom

    
    def f_exp(self, x, xi, y0):
        return y0*np.exp(x/xi)
    
    def compute_curve_fit_L1L(self):
        
        # curve fit for L1Lc
        xi_init = (self.Ls_vcg[-1]-self.Ls_vcg[0])/np.log(self.Kas_L1Lc[-1]/self.Kas_L1Lc[0])
        y0_init = self.Kas_L1Lc[-1] * np.exp(-self.Ls_vcg[-1]/xi_init)
        out = curve_fit(self.f_exp, self.Ls_vcg, self.Kas_L1Lc, p0=[xi_init, y0_init])[0]
        self.Lambda_L1Lc, self.Ka0_L1Lc = out

        # curve fit for L1Lf
        if np.all(np.isclose(np.mean(self.Kas_L1Lf), self.Kas_L1Lf)):
            self.Ka0_L1Lf = self.Kas_L1Lf[0]
        else:
            print("deviation!")

    def compute_curve_fit_L2L(self):

        # curve fit for L2Lc
        xi_init = (self.Ls_vcg[-1]-self.Ls_vcg[0])/np.log(self.Kas_L2Lc[-1]/self.Kas_L2Lc[0])
        y0_init = self.Kas_L2Lc[-1] * np.exp(-self.Ls_vcg[-1]/xi_init)
        out = curve_fit(self.f_exp, self.Ls_vcg, self.Kas_L2Lc, p0=[xi_init, y0_init])[0]
        self.Lambda_L2Lc, self.Ka0_L2Lc = out

        # curve fit for L2Lf
        xi_init = (self.Ls_vcg[-1]-self.Ls_vcg[0])/np.log(self.Kas_L2Lf[-1]/self.Kas_L2Lf[0])
        y0_init = self.Kas_L2Lf[-1] * np.exp(-self.Ls_vcg[-1]/xi_init)
        out = curve_fit(self.f_exp, self.Ls_vcg, self.Kas_L2Lf, p0=[xi_init, y0_init])[0]
        self.Lambda_L2Lf, self.Ka0_L2Lf = out

    def compute_curve_fits(self):
        self.compute_curve_fit_L1L()
        self.compute_curve_fit_L2L()

    def create_parameter_dictionary(self):
        self.params = {'Ka0_L1Lc':self.Ka0_L1Lc, \
                       'Ka0_L1Lf':self.Ka0_L1Lf, \
                       'Ka0_L2Lc':self.Ka0_L2Lc, \
                       'Ka0_L2Lf':self.Ka0_L2Lf}


ka_eff = EffectiveAssociationConstants(Ls_vcg=np.arange(4, 16, 1, dtype=int))
ka_eff.compute_effective_association_constants__all_Lsvcg()
ka_eff.compute_curve_fits()
ka_eff.create_parameter_dictionary()

f, axs = plt.subplots(1,2,figsize=(2*4.5,3.2),constrained_layout=True)

for i, letter in enumerate(['A', 'B']):
    axs[i].text(-0.13, 1.15, letter, transform=axs[i].transAxes, \
                fontsize=18, fontweight='bold', va='top', ha='right')

axs[0].scatter(ka_eff.Ls_vcg, ka_eff.Kas_L1Lc, facecolor='white', edgecolor='C0', marker='o', \
           label=r'$\frac{1\,|\,L}{L}$, c')
axs[0].plot(ka_eff.Ls_vcg_cont, ka_eff.f_exp(ka_eff.Ls_vcg_cont, \
                                             ka_eff.Lambda_L1Lc, ka_eff.Ka0_L1Lc), \
        color='C0')
axs[0].scatter(ka_eff.Ls_vcg, ka_eff.Kas_L1Lf, facecolor='white', edgecolor='C1', marker='o', \
           label=r'$\frac{1\,|\,L}{L}$, f')
axs[0].plot(ka_eff.Ls_vcg_cont, ka_eff.Ka0_L1Lf*np.ones(len(ka_eff.Ls_vcg_cont)), \
        color='C1')
axs[0].legend()
axs[0].set_yscale('log')
axs[0].set_xlabel('oligomer length $L_\mathrm{V}$')
axs[0].set_ylabel(r'$\mathcal{K}^a$ (1/M$^3$)')


axs[1].scatter(ka_eff.Ls_vcg, ka_eff.Kas_L2Lc, facecolor='white', edgecolor='C0', marker='o', \
           label=r'$\frac{2\,|\,L}{L}$, c')
axs[1].plot(ka_eff.Ls_vcg_cont, ka_eff.f_exp(ka_eff.Ls_vcg_cont, \
                                             ka_eff.Lambda_L2Lc, ka_eff.Ka0_L2Lc), \
        color='C0')
axs[1].scatter(ka_eff.Ls_vcg, ka_eff.Kas_L2Lf, facecolor='white', edgecolor='C1', marker='o', \
           label=r'$\frac{2\,|\,L}{L}$, f')
axs[1].plot(ka_eff.Ls_vcg_cont, ka_eff.f_exp(ka_eff.Ls_vcg_cont, \
                                             ka_eff.Lambda_L2Lf, ka_eff.Ka0_L2Lf), \
        color='C1')
axs[1].legend()
axs[1].set_yscale('log')
axs[1].set_xlabel('oligomer length $L_\mathrm{V}$')
axs[1].set_ylabel(r'$\mathcal{K}^a$ (1/M$^3$)')

plt.savefig(f'../../outputs/changing_LF/SI__scaling_effective_association_constants.pdf')