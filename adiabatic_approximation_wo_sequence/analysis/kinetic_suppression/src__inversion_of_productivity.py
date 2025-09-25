#!/bin/env python3

import sys
sys.path.append('../../../src/')
import os
from multiprocessing import Pool
import tqdm
from typing import Dict
from functools import partial
from scipy.optimize import root_scalar
from ComplexConstructor import *
from ConcentrationComputer import *

class DataPoint:

    def __init__(self, cmplxs, c0_stock, c0_vcg, cncs_params=None):

        self.cmplxs: ComplexConstructor = cmplxs
        if cncs_params == None:
            self.cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                                        c0_vcg_all_oligos=c0_vcg, Lambda_vcg=np.inf, \
                                        c0_stock_all_oligos=c0_stock, Lambda_stock=1/np.log(10))
        else:
            self.cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                                              c0_vcg_all_oligos=c0_vcg, \
                                              Lambda_vcg=cncs_params['Lambda_vcg'], \
                                              c0_stock_all_oligos=c0_stock, \
                                              Lambda_stock=cncs_params['Lambda_stock'])
        self.cncs.compute_equilibrium_concentration_log()
        self.cncs.compute_concentrations_productive_cvfolp()

    def compute_fraction_consumed_oligomers_via_extension_by_monomer(self):        
        
        self.fraction_consumed_oligos_by_monomer = {}
        for rt in self.cmplxs.react_type_cvfolp_2_index.keys():
            _, L1, L2 = eval(rt.split('_')[0])
            if L1 == 1 and L2 >= self.cmplxs.l_unique:
                i_L2 = self.cmplxs.key2indexcmplx[((L2,),())]
                if not L2 in self.fraction_consumed_oligos_by_monomer:
                    self.fraction_consumed_oligos_by_monomer[L2] \
                        = self.cncs.cs_cvfolp[rt]/self.cncs.cs_ss_comb_tot[i_L2]    
                else:
                    self.fraction_consumed_oligos_by_monomer[L2] \
                        += self.cncs.cs_cvfolp[rt]/self.cncs.cs_ss_comb_tot[i_L2]

    def compute_equilibrium_concentration_reactive_triplex(self):

        self.cs_triplexes_reactive_equ = {}
        for cmplx_key, cmplx_index in self.cmplxs.key2indexcmplx.items():
            if len(cmplx_key[0]) == 3:
                _, L1, L2 = cmplx_key[0]
                i, j = cmplx_key[1]
                if L1+i==j:
                    if not cmplx_key[0] in self.cs_triplexes_reactive_equ:
                        self.cs_triplexes_reactive_equ[cmplx_key[0]] = self.cncs.cs_cmplxs_equ[cmplx_index]
                    else:
                        self.cs_triplexes_reactive_equ[cmplx_key[0]] += self.cncs.cs_cmplxs_equ[cmplx_index]


class DataPoint__SingleLength(DataPoint):

    def __init__(self, cmplxs, c0_stock, c0_vcg, cncs_params=None):
        super().__init__(cmplxs=cmplxs, c0_stock=c0_stock, c0_vcg=c0_vcg, \
                         cncs_params=cncs_params)
        self.cncs_params = cncs_params
        self.cncs.compute_concentrations_added_nucleotides_productive_cvfolp()
        self.cncs.compute_yield_ratio_monomer_addition_exact()
        self.cncs.compute_fidelity_monomer_addition_exact()


class DataPoint__MultiLength(DataPoint):

    def __init__(self, cmplxs, c0_stock, c0_vcg, \
                 effective_association_constants__up_to_duplex, \
                 effective_association_constants__up_to_duplex__summed, \
                 effective_association_constants__reactive_triplexes, \
                 alphas, betas, betas_summed, \
                 number_iterations, cncs_params=None):
        super().__init__(cmplxs=cmplxs, c0_stock=c0_stock, c0_vcg=c0_vcg, \
                         cncs_params=cncs_params)
        self.cncs.compute_concentrations_productive_cvfolp()
        self.cncs.compute_yield_ratio_monomer_addition_exact()
        self.cncs.compute_fidelity_monomer_addition_exact()
        
        self.Lslong = [L for L in self.cmplxs.Ls if L >= self.cmplxs.l_unique]
        self.Llong2index = {L:i for i, L in enumerate(self.Lslong)}
        
        self.effective_association_constants__up_to_duplex \
            = effective_association_constants__up_to_duplex
        self.effective_association_constants__up_to_duplex__summed \
            = effective_association_constants__up_to_duplex__summed
        self.effective_association_constants__reactive_triplexes \
            = effective_association_constants__reactive_triplexes
        
        self.alphas = alphas
        self.betas = betas
        self.betas_summed = betas_summed

        self.number_iterations = number_iterations
    
    
    def compute_epsilons(self, cequs_tilde):

        epsilons = np.zeros((len(self.Lslong),len(self.Lslong)))
        for L, iL in self.Llong2index.items():
            for M, iM in self.Llong2index.items():
                iLl, iMl = self.cmplxs.L2indexss[L], self.cmplxs.L2indexss[M]
                epsilons[iL,iM] = cequs_tilde[iLl]-cequs_tilde[iMl]
        
        return epsilons

    
    def compute_equilibrium_concentration_single_strands__approximative_single_recursion(self, \
            epsilons):

        # uses the rescaling ctilde = sqrt( ctot/ (2*K_LL + sum_M K_LM))
        cequs_tilde = np.zeros(len(self.cmplxs.L2indexss))
        cequs = np.zeros(len(self.cmplxs.L2indexss))
        for L, iL in self.cmplxs.L2indexss.items():
            if L >= self.cmplxs.l_unique:
                ctot_L = self.cncs.cs_ss_comb_tot[iL]

                # compute effective alpha
                alpha = self.alphas[L]/np.sqrt(ctot_L)
                for M, iM in self.cmplxs.L2indexss.items():
                    if M >= self.cmplxs.l_unique:
                        iLl, iMl = self.Llong2index[L], self.Llong2index[M]
                        alpha += self.betas[iLl,iMl]*epsilons[iMl,iLl]

                # compute equilibrium concentration                
                cequ_tilde = (-alpha + np.sqrt(alpha**2 + 4*self.betas_summed[L]))/(2*self.betas_summed[L])
                cequs_tilde[iL] = cequ_tilde

                # for new choice of alpha and beta
                cequ = np.sqrt( ctot_L / (2*self.effective_association_constants__up_to_duplex[(L,L)] + \
                                            + self.effective_association_constants__up_to_duplex__summed[L]) ) \
                        * cequ_tilde
                cequs[iL] = cequ
        
        return cequs_tilde, cequs
    
    
    def compute_equilibrium_concentration_single_strands__approximative(self):

        self.iteration_2_cs_ss_equ_approx = {}
        self.iteration_2_cs_tilde_ss_equ_approx = {}
        n = 0
        epsilons = np.zeros((len(self.Lslong),len(self.Lslong)))
        while n < self.number_iterations:
            cequs_tilde, cequs = \
                self.compute_equilibrium_concentration_single_strands__approximative_single_recursion(\
                    epsilons)
            epsilons = self.compute_epsilons(cequs_tilde)
            self.iteration_2_cs_tilde_ss_equ_approx[n] = cequs_tilde
            self.iteration_2_cs_ss_equ_approx[n] = cequs
            n += 1



class DataSet:

    def __init__(self, L_vcg_min, L_vcg_max, c0_stock, ratios_cvcgtot_cstocktot, \
                 cmplxs_params=None, cncs_params=None):

        if cmplxs_params==None:
            self.cmplxs = ComplexConstructor(l_unique=3, alphabet=4, \
                                L_vcg=L_vcg_min, L_vcg_min=L_vcg_min, L_vcg_max=L_vcg_max, \
                                Lmax_stock=2, comb_vcg=32, gamma_2m=-2.5, gamma_d=-2.5/2, \
                                include_triplexes=True, include_tetraplexes=True)
        else:
            self.cmplxs = ComplexConstructor(l_unique=cmplxs_params['l_unique'], \
                                             alphabet=cmplxs_params['alphabet'], \
                                             L_vcg=L_vcg_min, L_vcg_min=L_vcg_min, L_vcg_max=L_vcg_max, \
                                             Lmax_stock=cmplxs_params['Lmax_stock'], \
                                             comb_vcg=cmplxs_params['comb_vcg'], \
                                             gamma_2m=cmplxs_params['gamma_2m'], \
                                             gamma_d=cmplxs_params['gamma_d'], \
                                             include_triplexes=cmplxs_params['include_triplexes'], \
                                             include_tetraplexes=cmplxs_params['include_tetraplexes'])
        self.cmplxs.identify_reaction_types_cvfolp_all_complexes()
        self.cncs_params = cncs_params

        self.c0_stock = c0_stock
        self.ratios_cvcgtot_cstocktot = ratios_cvcgtot_cstocktot


    def compute_effective_association_constants__up_to_duplex(self):
        
        if self.cncs_params == None:
            cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                                         c0_vcg_all_oligos=None, Lambda_vcg=np.inf, \
                                         c0_stock_all_oligos=None, Lambda_stock=1/np.log(10))
        else:
            cncs = ConcentrationComputer(cmplx=self.cmplxs, \
                                         c0_vcg_all_oligos=None, \
                                         Lambda_vcg=self.cncs_params['Lambda_vcg'], \
                                         c0_stock_all_oligos=None, \
                                         Lambda_stock=self.cncs_params['Lambda_stock'])
        cncs.compute_ratio_concentration_strand_to_reference_strand()
        cncs.compute_complex_weights()

        self.effective_association_constants__up_to_duplex = {}
        for cmplx_key, cmplx_index in self.cmplxs.key2indexcmplx.items():
            if len(cmplx_key[0]) <= 2:        
                if not cmplx_key[0] in self.effective_association_constants__up_to_duplex:
                    self.effective_association_constants__up_to_duplex[cmplx_key[0]] = cncs.ws_cmplxs[cmplx_index]
                else:
                    self.effective_association_constants__up_to_duplex[cmplx_key[0]] += cncs.ws_cmplxs[cmplx_index]

    def compute_effective_association_constants__reactive_triplex(self):

        if self.cncs_params == None:
            cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                                         c0_vcg_all_oligos=None, Lambda_vcg=np.inf, \
                                         c0_stock_all_oligos=None, Lambda_stock=1/np.log(10))
        else:
            cncs = ConcentrationComputer(cmplx=self.cmplxs, \
                                         c0_vcg_all_oligos=None, \
                                         Lambda_vcg=self.cncs_params['Lambda_vcg'], \
                                         c0_stock_all_oligos=None, \
                                         Lambda_stock=self.cncs_params['Lambda_stock'])
        cncs.compute_ratio_concentration_strand_to_reference_strand()
        cncs.compute_complex_weights()
        
        self.effective_association_constants__reactive_triplexes = {}
        for cmplx_key, cmplx_index in self.cmplxs.key2indexcmplx.items():
            if len(cmplx_key[0]) == 3:
                _, L1, _ = cmplx_key[0]
                i, j = cmplx_key[1]
                if L1+i==j:
                    if not cmplx_key[0] in self.effective_association_constants__reactive_triplexes:
                        self.effective_association_constants__reactive_triplexes[cmplx_key[0]] = cncs.ws_cmplxs[cmplx_index]
                    else:
                        self.effective_association_constants__reactive_triplexes[cmplx_key[0]] += cncs.ws_cmplxs[cmplx_index]



class DataSet__SingleLength(DataSet):

    def __init__(self, L_vcg, c0_stock, ratios_cvcgtot_cstocktot, \
                 cmplxs_params=None, cncs_params=None):
        
        super().__init__(L_vcg_min=L_vcg, L_vcg_max=L_vcg, c0_stock=c0_stock, \
                         ratios_cvcgtot_cstocktot=ratios_cvcgtot_cstocktot, \
                         cmplxs_params=cmplxs_params, cncs_params=cncs_params)
        self.L_vcg = L_vcg
        self.cncs_params = cncs_params
    
    def compute_asymptotic_fraction_consumed_oligos_by_monomer(self):

        if self.cncs_params == None:
            cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                                         c0_vcg_all_oligos=1., Lambda_vcg=np.inf, \
                                         c0_stock_all_oligos=self.c0_stock, Lambda_stock=1/np.log(10))
        else:
            cncs = ConcentrationComputer(cmplx=self.cmplxs, \
                                         c0_vcg_all_oligos=1., \
                                         Lambda_vcg=self.cncs_params['Lambda_vcg'], \
                                         c0_stock_all_oligos=self.c0_stock, \
                                         Lambda_stock=self.cncs_params['Lambda_stock'])
        c1tot = cncs.cs_ss_tot[0]
        self.fraction_consumed_oligos_by_monomer__asymptotic = \
            2/3 * (1 + 1/np.exp(1.25-2.5) - 2*np.exp(1.25-2.5)/3) * c1tot
    
    def compute_total_inversion_concentration(self, f):
        
        self.c0_vcg_threshold = self.effective_association_constants__up_to_duplex[(self.L_vcg,)]**2 \
                                / (2* self.effective_association_constants__up_to_duplex[(self.L_vcg,self.L_vcg)]) \
                                * (f)/(1-f)**2
    
    def compute_single_data_point(self, ratio_cvcgtot_cstocktot):

        dp = DataPoint__SingleLength(cmplxs=self.cmplxs, c0_stock=self.c0_stock, \
                                     c0_vcg=self.c0_stock*ratio_cvcgtot_cstocktot, \
                                     cncs_params=self.cncs_params)
        dp.compute_fraction_consumed_oligomers_via_extension_by_monomer()

        cs_ss_equ = dp.cncs.cs_ss_equ
        cs_ss_comb_equ = dp.cncs.cs_cmplxs_equ[0:len(self.cmplxs.Ls)]
        
        cs_ss_tot = dp.cncs.cs_ss_tot
        cs_ss_comb_tot = dp.cncs.cs_ss_comb_tot
        
        fraction_consumed_oligos_by_monomer = dp.fraction_consumed_oligos_by_monomer

        yield_ratio = dp.cncs.yield_ratio_monomer_addition_exact
        fidelity = dp.cncs.fidelity_monomer_addition_exact

        return cs_ss_tot, cs_ss_comb_tot, \
               cs_ss_equ, cs_ss_comb_equ, \
               fraction_consumed_oligos_by_monomer, \
               yield_ratio, fidelity
    
    def compute_all_data_points(self):

        outs = []
        for ratio_cvcgtot_cstocktot in self.ratios_cvcgtot_cstocktot:
            out = self.compute_single_data_point(ratio_cvcgtot_cstocktot)
            outs.append(out) 

        self.css_ss_tot = {L:[] for L in self.cmplxs.Ls}
        self.css_ss_comb_tot = {L:[] for L in self.cmplxs.Ls}

        self.css_ss_equ = {L:[] for L in self.cmplxs.Ls}
        self.css_ss_comb_equ = {L:[] for L in self.cmplxs.Ls}

        self.fractions_consumed_oligos_by_monomer = {self.L_vcg:[]}

        self.yields = []
        self.fidelities = []
        self.efficiencies = []
        
        for out in outs:

            cs_ss_tot, cs_ss_comb_tot, \
            cs_ss_equ, cs_ss_comb_equ, \
            fraction_consumed_oligos_by_monomer, \
            yield_ratio, fidelity = out

            for iL, L in enumerate(self.cmplxs.Ls):

                self.css_ss_tot[L].append(cs_ss_tot[iL])
                self.css_ss_comb_tot[L].append(cs_ss_comb_tot[iL])
                
                self.css_ss_equ[L].append(cs_ss_equ[iL])
                self.css_ss_comb_equ[L].append(cs_ss_comb_equ[iL])
            
                if L >= self.cmplxs.l_unique:
                    self.fractions_consumed_oligos_by_monomer[L].append(fraction_consumed_oligos_by_monomer[L])
            
            self.yields.append(yield_ratio)
            self.fidelities.append(fidelity)
            self.efficiencies.append(yield_ratio*fidelity)


        self.css_ss_tot = {key:np.asarray(values) for key, values in self.css_ss_tot.items()}
        self.css_ss_comb_tot = {key:np.asarray(values) \
                                for key, values in self.css_ss_comb_tot.items()}

        self.css_ss_equ = {key:np.asarray(values) for key, values in self.css_ss_equ.items()}
        self.css_ss_comb_equ = {key:np.asarray(values) \
                                for key, values in self.css_ss_comb_equ.items()}

        self.fractions_consumed_oligos_by_monomer = \
            {key:np.asarray(values) \
             for key, values in self.fractions_consumed_oligos_by_monomer.items()}
        
        self.yields = np.asarray(self.yields)
        self.fidelities = np.asarray(self.fidelities)


class DataSet__MultiLength(DataSet):

    def __init__(self, L_vcg_min, L_vcg_max, c0_stock, ratios_cvcgtot_cstocktot, \
                 number_iterations, Ls_vcg_sets, cmplxs_params=None):
        
        super().__init__(L_vcg_min=L_vcg_min, L_vcg_max=L_vcg_max, c0_stock=c0_stock, \
                         ratios_cvcgtot_cstocktot=ratios_cvcgtot_cstocktot, \
                         cmplxs_params=cmplxs_params)
        self.Lslong = [L for L in self.cmplxs.Ls if L >= self.cmplxs.l_unique]
        self.Llong2index = {L:i for i, L in enumerate(self.Lslong)}

        # properties of the approximation
        self.number_iterations = number_iterations
        self.Ls_vcg_sets = Ls_vcg_sets
        
    def compute_alphas_and_betas_for_approximate_equilibration(self):
        self.compute_alphas_and_betas_for_approximate_equilibration__new()
        
    def compute_alphas_and_betas_for_approximate_equilibration__conventional(self):

        if not hasattr(self, 'effective_association_constants'):
            self.compute_effective_association_constants__up_to_duplex()

        self.effective_association_constants__up_to_duplex__summed = None

        self.betas = np.zeros((len(self.Lslong),len(self.Lslong)))
        for L1, i1 in self.Llong2index.items():
            for L2, i2 in self.Llong2index.items():
                Lshort, Llong = min([L1,L2]), max([L1,L2])
                if L1 != L2:
                    beta = self.effective_association_constants__up_to_duplex[(Llong,Lshort)] \
                            /np.sqrt(4 * self.effective_association_constants__up_to_duplex[(L1,L1)] \
                                     * self.effective_association_constants__up_to_duplex[(L2,L2)])
                    self.betas[i1,i2] = beta
                    self.betas[i2,i1] = beta
                elif L1 == L2:
                    beta = 1
                    self.betas[i1,i2] = beta

        self.alphas = {L:self.effective_association_constants__up_to_duplex[(L,)] \
                         /np.sqrt(2*self.effective_association_constants__up_to_duplex[(L,L)]) \
                       for L in self.Lslong}
        self.betas_summed = {L:np.sum(self.betas[self.Llong2index[L]]) for L in self.Lslong}

    def compute_alphas_and_betas_for_approximate_equilibration__new(self):

        if not hasattr(self, 'effecitve_association_constants'):
            self.compute_effective_association_constants__up_to_duplex()
        
        self.effective_association_constants__up_to_duplex__summed = {L:0. for L in self.Lslong}
        for L1, i1 in self.Llong2index.items():
            for L2, i2 in self.Llong2index.items():
                Lshort, Llong = min([L1,L2]), max([L1,L2])
                self.effective_association_constants__up_to_duplex__summed[L1] \
                    += self.effective_association_constants__up_to_duplex[(Llong,Lshort)]
        
        self.betas = np.zeros((len(self.Lslong), len(self.Lslong)))
        for L1, i1 in self.Llong2index.items():
            for L2, i2 in self.Llong2index.items():
                Lshort, Llong = min([L1,L2]), max([L1,L2])
                if L1 != L2:
                    beta = self.effective_association_constants__up_to_duplex[(Llong,Lshort)] \
                           / np.sqrt( (2*self.effective_association_constants__up_to_duplex[(L1,L1)] \
                                       + self.effective_association_constants__up_to_duplex__summed[L1]) \
                                    * (2*self.effective_association_constants__up_to_duplex[(L2,L2)] \
                                       + self.effective_association_constants__up_to_duplex__summed[L2]) )
                    self.betas[i1,i2] = beta
                    self.betas[i2,i1] = beta
                if L1 == L2:
                    beta = (2*self.effective_association_constants__up_to_duplex[(L1,L1)]) \
                           / ( 2*self.effective_association_constants__up_to_duplex[(L1,L1)] \
                               + self.effective_association_constants__up_to_duplex__summed[L1] )
                    self.betas[i1,i1] = beta
        
        self.alphas = {L:self.effective_association_constants__up_to_duplex[(L,)] \
                         / np.sqrt( 2*self.effective_association_constants__up_to_duplex[(L,L)] \
                                    + self.effective_association_constants__up_to_duplex__summed[L]) \
                       for L in self.Lslong}
        self.betas_summed = {L:np.sum(self.betas[self.Llong2index[L]]) for L in self.Lslong}
    
    def compute_epsilons(self, cequs_tilde):

        epsilons = np.zeros((len(self.Lslong),len(self.Lslong)))
        for L, iL in self.Llong2index.items():
            for M, iM in self.Llong2index.items():
                iLl, iMl = self.cmplxs.L2indexss[L], self.cmplxs.L2indexss[M]
                epsilons[iL,iM] = cequs_tilde[iLl]-cequs_tilde[iMl]
        
        return epsilons
    
    def compute_equilibrium_concentration_single_strands__approximative_single_recursion(self, \
            ctot, epsilons):

        # uses the rescaling ctilde = sqrt( ctot/ (2*K_LL + sum_M K_LM)) -> this is used in the paper
        cequs_tilde = np.zeros(len(self.cmplxs.L2indexss))
        cequs = np.zeros(len(self.cmplxs.L2indexss))
        for L, iL in self.cmplxs.L2indexss.items():
            if L >= self.cmplxs.l_unique:
                # compute effective alpha
                alpha = self.alphas[L]/np.sqrt(ctot)
                for M, iM in self.cmplxs.L2indexss.items():
                    if M >= self.cmplxs.l_unique:
                        iLl, iMl = self.Llong2index[L], self.Llong2index[M]
                        alpha += self.betas[iLl,iMl]*epsilons[iMl,iLl]

                # compute equilibrium concentration                
                cequ_tilde = (-alpha + np.sqrt(alpha**2 + 4*self.betas_summed[L]))/(2*self.betas_summed[L])
                cequs_tilde[iL] = cequ_tilde

                # for new ("hebrew") choice of alpha and beta
                cequ = np.sqrt( ctot / (2*self.effective_association_constants__up_to_duplex[(L,L)] + \
                                            + self.effective_association_constants__up_to_duplex__summed[L]) ) * cequ_tilde
                cequs[iL] = cequ
        
        return cequs_tilde, cequs
    
    def compute_equilibrium_concentration_single_strands__approximative(self, ctot):

        n = 0
        epsilons = np.zeros((len(self.Lslong),len(self.Lslong)))
        while n < self.number_iterations:
            cequs_tilde, cequs = \
                self.compute_equilibrium_concentration_single_strands__approximative_single_recursion(\
                    ctot=ctot, epsilons=epsilons)
            epsilons = self.compute_epsilons(cequs_tilde)
            n += 1
        
        return cequs, cequs_tilde
    
    def compute_difference_Lreactive_Mreactive__single_pair(self, ctot, L, M):

        cequs, _ = self.compute_equilibrium_concentration_single_strands__approximative(\
            ctot=ctot)
        
        iL = self.cmplxs.L2indexss[L]
        iM = self.cmplxs.L2indexss[M]

        diff = 0.
        for N in self.Lslong:
            iN = self.cmplxs.L2indexss[N]
            diff += self.effective_association_constants__reactive_triplexes[(N,1,L)] \
                    * cequs[iN]*cequs[iL]
        for P in self.Lslong:
            iP = self.cmplxs.L2indexss[P]
            diff -= self.effective_association_constants__reactive_triplexes[(P,1,M)] \
                    * cequs[iP]*cequs[iM]
            
        return diff
    
    def compute_total_inversion_concentration__single_pair(self, L, M):
        f = partial(\
                self.compute_difference_Lreactive_Mreactive__single_pair, L=L, M=M)
        out = root_scalar(f, bracket=[1e-10*self.c0_stock, 2*self.c0_stock], \
                          method='brentq', xtol=1e-15, rtol=1e-15)
        ctotL = out.root
        # convert output concentration (per single VCG strand) to total VCG concentration
        ctot = ctotL*(self.cmplxs.L_vcg_max-self.cmplxs.L_vcg_min+1)

        return ctot

    def compute_total_inversion_concentration__all_pairs(self):
        self.c0_vcg_thresholds = {}
        for Ls_vcg_set in self.Ls_vcg_sets:
            self.c0_vcg_thresholds[Ls_vcg_set] = \
                self.compute_total_inversion_concentration__single_pair(Ls_vcg_set[0], Ls_vcg_set[1])

    def compute_single_data_point(self, ratio_cvcgtot_cstocktot):

        dp = DataPoint__MultiLength(cmplxs=self.cmplxs, c0_stock=self.c0_stock, \
                       c0_vcg=self.c0_stock*ratio_cvcgtot_cstocktot, \
                       effective_association_constants__up_to_duplex=\
                        self.effective_association_constants__up_to_duplex, \
                       effective_association_constants__up_to_duplex__summed=\
                        self.effective_association_constants__up_to_duplex__summed, \
                       effective_association_constants__reactive_triplexes=None, \
                       alphas=self.alphas, betas=self.betas, betas_summed=self.betas_summed, \
                       number_iterations=self.number_iterations)
        dp.compute_equilibrium_concentration_single_strands__approximative()
        dp.compute_fraction_consumed_oligomers_via_extension_by_monomer()
        
        cs_ss_equ = dp.cncs.cs_ss_equ
        cs_ss_comb_equ = dp.cncs.cs_cmplxs_equ[0:len(self.cmplxs.Ls)]
        
        cs_ss_tot = dp.cncs.cs_ss_tot
        cs_ss_comb_tot = dp.cncs.cs_ss_comb_tot
        
        cs_ss_equ__approx0 = dp.iteration_2_cs_ss_equ_approx[0]
        cs_ss_comb_equ__approx0 = dp.cmplxs.combs[0:len(self.cmplxs.Ls)]*cs_ss_equ__approx0

        cs_ss_equ__approx1 = dp.iteration_2_cs_ss_equ_approx[1]
        cs_ss_comb_equ__approx1 = dp.cmplxs.combs[0:len(self.cmplxs.Ls)]*cs_ss_equ__approx1

        fraction_consumed_oligos_by_monomer = dp.fraction_consumed_oligos_by_monomer

        cs_cvfolp = dp.cncs.cs_cvfolp

        yield_ratio = dp.cncs.yield_ratio_monomer_addition_exact
        fidelity = dp.cncs.fidelity_monomer_addition_exact
        efficiency = dp.cncs.yield_ratio_monomer_addition_exact * dp.cncs.fidelity_monomer_addition_exact

        # compute relative mean squared error
        rmses = {}
        for iteration in dp.iteration_2_cs_ss_equ_approx.keys():
            i_cutoff = self.cmplxs.L2indexss[self.cmplxs.l_unique]
            rmse = np.sqrt(np.mean((dp.cncs.cs_ss_equ[i_cutoff:] \
                                    - dp.iteration_2_cs_ss_equ_approx[iteration][i_cutoff:])**2\
                                    /dp.cncs.cs_ss_equ[i_cutoff:]**2))
            rmses[iteration] = rmse
        
        del dp

        return cs_ss_tot, cs_ss_comb_tot, \
               cs_ss_equ, cs_ss_comb_equ, \
               cs_ss_equ__approx0, cs_ss_comb_equ__approx0, \
               cs_ss_equ__approx1, cs_ss_comb_equ__approx1, \
               rmses, fraction_consumed_oligos_by_monomer, cs_cvfolp, efficiency

    def compute_all_data_points(self):

        p = Pool(os.cpu_count()-4)
        outs = []
        for out in tqdm.tqdm(p.imap(self.compute_single_data_point, \
                                    self.ratios_cvcgtot_cstocktot), \
                             total=len(self.ratios_cvcgtot_cstocktot)):
            outs.append(out)
        p.close()
        
        # outs = []
        # for ratio_cvcgtot_cstocktot in self.ratios_cvcgtot_cstocktot:
        #     out = self.compute_single_data_point(ratio_cvcgtot_cstocktot)
        #     outs.append(out) 

        self.css_ss_tot = {L:[] for L in self.cmplxs.Ls}
        self.css_ss_comb_tot = {L:[] for L in self.cmplxs.Ls}

        self.css_ss_equ = {L:[] for L in self.cmplxs.Ls}
        self.css_ss_comb_equ = {L:[] for L in self.cmplxs.Ls}

        self.css_ss_equ__approx0 = {L:[] for L in self.cmplxs.Ls}
        self.css_ss_comb_equ__approx0 = {L:[] for L in self.cmplxs.Ls}

        self.css_ss_equ__approx1 = {L:[] for L in self.cmplxs.Ls}
        self.css_ss_comb_equ__approx1 = {L:[] for L in self.cmplxs.Ls}

        self.rmses = {iteration:[] for iteration in range(self.number_iterations)}
        
        self.fractions_consumed_oligos_by_monomer = {L:[] for L in self.Lslong}
        self.css_cvfolp =  {key:[] for key in self.cmplxs.react_type_cvfolp_2_index.keys()}

        self.efficiencies = []

        for out in outs:

            cs_ss_tot, cs_ss_comb_tot, \
            cs_ss_equ, cs_ss_comb_equ, \
            cs_ss_equ__approx0, cs_ss_comb_equ__approx0, \
            cs_ss_equ__approx1, cs_ss_comb_equ__approx1, \
            rmses_loc, fraction_consumed_oligos_by_monomer, cs_cvfolp, efficiency = out

            for iL, L in enumerate(self.cmplxs.Ls):

                self.css_ss_tot[L].append(cs_ss_tot[iL])
                self.css_ss_comb_tot[L].append(cs_ss_comb_tot[iL])
                
                self.css_ss_equ[L].append(cs_ss_equ[iL])
                self.css_ss_comb_equ[L].append(cs_ss_comb_equ[iL])

                self.css_ss_equ__approx0[L].append(cs_ss_equ__approx0[iL])
                self.css_ss_comb_equ__approx0[L].append(cs_ss_comb_equ__approx0[iL])

                self.css_ss_equ__approx1[L].append(cs_ss_equ__approx1[iL])
                self.css_ss_comb_equ__approx1[L].append(cs_ss_comb_equ__approx1[iL])

                if L >= self.cmplxs.l_unique:
                    self.fractions_consumed_oligos_by_monomer[L].append(fraction_consumed_oligos_by_monomer[L])

            for iteration in range(self.number_iterations):
                self.rmses[iteration].append(rmses_loc[iteration])
            
            for rt in self.cmplxs.react_type_cvfolp_2_index.keys():
                self.css_cvfolp[rt].append(cs_cvfolp[rt])

            self.efficiencies.append(efficiency)
        
        self.css_ss_tot = {key:np.asarray(values) for key, values in self.css_ss_tot.items()}
        self.css_ss_comb_tot = {key:np.asarray(values) \
                                for key, values in self.css_ss_comb_tot.items()}

        self.css_ss_equ = {key:np.asarray(values) for key, values in self.css_ss_equ.items()}
        self.css_ss_comb_equ = {key:np.asarray(values) \
                                for key, values in self.css_ss_comb_equ.items()}

        self.css_ss_equ__approx0 = {key:np.asarray(values) \
                                    for key, values in self.css_ss_equ__approx0.items()}
        self.css_ss_comb_equ__approx0 = {key:np.asarray(values) \
                                            for key, values in self.css_ss_comb_equ__approx0.items()}

        self.css_ss_equ__approx1 = {key:np.asarray(values) \
                                    for key, values in self.css_ss_equ__approx1.items()}
        self.css_ss_comb_equ__approx1 = {key:np.asarray(values) \
                                            for key, values in self.css_ss_comb_equ__approx1.items()}

        self.rmses = {key:np.mean(values) for key, values in self.rmses.items()}
        
        self.fractions_consumed_oligos_by_monomer = \
            {key:np.asarray(values) \
             for key, values in self.fractions_consumed_oligos_by_monomer.items()}
        self.css_cvfolp = {key:np.asarray(values) for key, values in self.css_cvfolp.items()}

        self.efficiencies = np.asarray(self.efficiencies)
