#!/bin/env python3

import sys
sys.path.append('../../src/')
import os

from ComplexConstructor import *
from ConcentrationComputer import *

class DatasetTheo:

    def __init__(self, L_vcg, gamma_2m, \
                 c0_stock_all, c0_vcg_all, optimize, compute_overlap):

        self.cmplxs = ComplexConstructor(\
            l_unique=3, alphabet=4, \
            L_vcg=L_vcg, L_vcg_min=L_vcg, L_vcg_max=L_vcg, \
            Lmax_stock=1, comb_vcg=32, gamma_2m=gamma_2m, gamma_d=gamma_2m/2, \
            include_tetraplexes=True)
        
        if (not optimize):
            self.cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                            c0_vcg_all_oligos=c0_vcg_all, Lambda_vcg=np.inf, \
                            c0_stock_all_oligos=c0_stock_all, Lambda_stock=np.inf)
            self.cncs.compute_equilibrium_concentration_log()
            self.cncs.compute_concentrations_added_nucleotides_productive_cvflvs()
            self.cncs.compute_errorfree_ratio_cvflvs_exact()
            self.cncs.compute_yield_ratio_cvflvs_exact()
            # self.ratio_templig_to_fidelity = self.cncs.cs_nuc_cvflvs['vv_v_v_c']/self.cncs.errorfree_ratio_cvflvs_exact
        
        elif optimize:
            self.cncs = ConcentrationOptimizerAddedNucsExact(cmplxs=self.cmplxs, \
                            Lambda_vcg=np.inf, Lambda_stock=np.inf, \
                            c0_stock_all_oligos=c0_stock_all)
            self.cncs.compute_optimal_total_concentration_ratio_cvflvs_exact()
            self.cncs.compute_optimal_equilibrium_concentration_ratio_cvflvs_exact()

            self.cncs.compute_close_to_optimal_total_concentration_ratio_cvflvs()
            self.cncs.compute_close_to_optimal_equilibrium_concentration_ratio_cvflvs()

class AnalyticPrediction:

    def __init__(self, K_DI_0, m_DI, \
                 K_PE_0, L_PE, \
                 K_TLC_0, L_TLC, \
                 K_TL_0, L_TL, Ls):

        self.K_DI_0 = K_DI_0
        self.m_DI = m_DI

        self.K_PE_0 = K_PE_0
        self.L_PE = L_PE
    
        self.K_TLC_0 = K_TLC_0
        self.L_TLC = L_TLC

        self.K_TL_0 = K_TL_0
        self.L_TL = L_TL

        self.L_TL_eff = (self.L_TLC + self.L_TL)/2
    
        self.Ls = Ls
    
    def compute_optimal_equilibrium_concentration_ratios(self):
        
        part1 = np.sqrt( (self.K_DI_0 + self.m_DI*self.Ls)/(self.K_TL_0 - self.K_TLC_0) ) \
                * np.exp(-self.Ls/(2*self.L_TL_eff))
        part2 = ((self.K_DI_0+self.m_DI*self.Ls)*self.K_TLC_0)/(self.K_PE_0*(self.K_TL_0-self.K_TLC_0)) * np.exp(-self.Ls/self.L_PE)
        self.rs_opt = 8*(part1 + part2) # the 8 is a combinatoric prefactor
    
    def compute_optimal_fidelity(self):
        self.f0 = (self.K_TL_0 - self.K_TLC_0 + self.m_DI)/self.K_PE_0
        self.fs_opt = 1 - self.f0*np.sqrt(self.Ls)*np.exp(- (1/self.L_PE-1/(2*self.L_TL_eff)) * self.Ls)

    def compute_characteristic_lenghtscale(self):
        f0 = (self.K_TL_0 - self.K_TLC_0 + self.m_DI)/self.K_PE_0
        L_eff = 1/(1/self.L_PE-1/(2*self.L_TL_eff))
        compute_diff = lambda L: f0*np.exp(-L/L_eff) - 2e-2
        self.L_c = root_scalar(compute_diff, bracket=[2*L_eff, 100*L_eff]).root


class AnalyticPredictionNew:

    def __init__(self, params, Ls):
        
        self.Ka0_FF_all = params['Ka0_FF_all']
        self.Lambda_FF_all = params['Lambda_FF_all']
        
        self.Ka0_FV_all = params['Ka0_FV_all']
        self.Lambda_FV_all = params['Lambda_FV_all']
        self.Ka0_FV_corr = params['Ka0_FV_corr']
        self.Lambda_FV_corr = params['Lambda_FV_corr']
        
        self.Ka0_VV_all = params['Ka0_VV_all']
        self.Lambda_VV_all = params['Lambda_VV_all']
        self.Ka0_VV_corr = params['Ka0_VV_corr']
        self.Lambda_VV_corr = params['Lambda_VV_corr']
        self.Ka0_VV_err = params['Ka0_VV_err']
        self.Lambda_VV_err = params['Lambda_VV_err']

        self.Ls = Ls
    
    def compute_optimal_equilibrium_concentration_ratios(self):

        # self.rs_opt = 8 * np.sqrt(self.Ka0_FF_all/self.Ka0_VV_err) \
        #               * np.sqrt( 1/self.Lambda_FF_all - 1/self.Ls) * np.exp(-self.Ls/(2*self.Lambda_VV_err))
        
        Ka_FF_all = self.Ka0_FF_all*(self.Ls/self.Lambda_FF_all - 1)
        Ka_FV_all = self.Ka0_FV_all*np.exp(self.Ls/self.Lambda_FV_all)
        Ka_FV_corr = self.Ka0_FV_corr*np.exp(self.Ls/self.Lambda_FV_corr)
        Ka_VV_corr = self.Ka0_VV_corr*self.Ls*np.exp(self.Ls/self.Lambda_VV_corr)
        Ka_VV_all = self.Ka0_VV_all*np.exp(self.Ls/self.Lambda_VV_all)
        part1 = Ka_FF_all*Ka_VV_corr/(Ka_VV_all*Ka_FV_corr - Ka_VV_corr*Ka_FV_all)
        part2_num = np.sqrt( self.Ls**2 * Ka_FF_all**2 * Ka_VV_corr**2 \
                            + self.Ls * Ka_FF_all * Ka_FV_corr * (Ka_VV_all*Ka_FV_corr - Ka_VV_corr*Ka_FV_all))
        part2_denom = self.Ls * (Ka_VV_all*Ka_FV_corr - Ka_VV_corr*Ka_FV_all)
        self.rs_opt = 8*(part1 + part2_num/part2_denom)

    def compute_maximal_efficiencies(self):
        f0 = 2*np.sqrt(self.Ka0_FF_all*self.Ka0_VV_err)/self.Ka0_FV_all
        compute_diff = lambda L: f0*np.sqrt(1/self.Lambda_FF_all - 1/L)*L*np.exp(-L/(2*self.Lambda_VV_err)) - 3e-1
        Lcut = root_scalar(compute_diff, bracket=[3, 20]).root
        self.Ls_etas = np.linspace(Lcut, self.Ls[-1], 50)
        self.etas = 1 - f0*np.sqrt(1/self.Lambda_FF_all - 1/self.Ls_etas)*self.Ls_etas*np.exp(-self.Ls_etas/(2*self.Lambda_VV_err))

    def compute_characteristic_lengthscale(self):
        f0 = 2*np.sqrt(self.Ka0_FF_all*self.Ka0_VV_err)/self.Ka0_FV_all
        compute_diff = lambda L: f0*np.sqrt(1/self.Lambda_FF_all - 1/L)*L*np.exp(-L/(2*self.Lambda_VV_err)) - 5e-2
        self.L_c = root_scalar(compute_diff, bracket=[3, 20]).root
