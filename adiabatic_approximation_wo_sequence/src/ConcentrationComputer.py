#!/bin/env python3

from scipy.optimize import root, root_scalar, minimize_scalar, minimize
import warnings
from numba import njit

from ComplexConstructor import *
import helper_functions as hf

@njit
def compute_complex_concentrations_log_njit(cs_ss_log, n_cmplxs, \
                                            indexcmplx2indicesstrands_arr, \
                                            combs_log, Kds_log):

    cs_cmplxs_log = np.zeros(n_cmplxs)

    for i_cmplx, is_strands in enumerate(indexcmplx2indicesstrands_arr):
        is_strands_eff = is_strands[is_strands >= 0]
        cs_cmplxs_log[i_cmplx] = combs_log[i_cmplx] \
                                 + np.sum(cs_ss_log[is_strands_eff]) \
                                 - Kds_log[i_cmplx]
        
    return np.exp(cs_cmplxs_log)


@njit
def compute_mass_conservation_log_njit(cs_ss_log, n_cmplxs, \
                                  indexcmplx2indicesstrands_arr, \
                                  indexstrand2indicescmplxs_arr, \
                                  combs_log, Kds_log, cs_ss_comb_tot):

    cs_cmplxs = compute_complex_concentrations_log_njit(cs_ss_log, n_cmplxs, \
                                                indexcmplx2indicesstrands_arr, \
                                                combs_log, Kds_log)
    
    diff = np.zeros(len(cs_ss_log))

    for i_strand, is_cmplxs in enumerate(indexstrand2indicescmplxs_arr):
        is_cmplxs_eff = is_cmplxs[is_cmplxs >= 0]
        diff[i_strand] = cs_ss_comb_tot[i_strand] - np.sum(cs_cmplxs[is_cmplxs_eff])

    return diff


class ConcentrationComputer:

    def __init__(self, cmplxs: ComplexConstructor, \
                 c0_vcg_single_oligo=None, c0_vcg_all_oligos=None, Lambda_vcg=None, \
                 c0_stock_single_oligo=None, c0_stock_all_oligos=None, Lambda_stock=None, 
                 cs_ss_comb_tot=None, optional_parameters=None):

        # container of all possible complexes
        self.cmplxs = cmplxs

        # concentrations can be given via two ways:
        # i) c0_?_single_oligo: every concentration is measured per type of strand
        #    i. e. the monomer concentration is the concentration for one monomer 
        #    (e. g. A), to compute the concentration of all monomers one has to 
        #    multiply by the combinatoric multiplicity
        # ii) c0_?_all_oligos: concentration of all oligomers contained in the respective
        #    subgroup of strands, e. g. all strands belonging to the VCG or the feeding stock
        # to convert between single_oligo and all_oligos representation one has
        # to divide/multiply by the combinatoric prefactor and the exponential
        # surpression factor

        # total concentrations of oligomers of length L_vcg per type
        self.c0_vcg_single_oligo = c0_vcg_single_oligo
        self.c0_vcg_all_oligos = c0_vcg_all_oligos
        # characteristic length scale of exponential decay in oligomer length 
        # around L_vcg peak
        self.Lambda_vcg = Lambda_vcg 

        # total concentration of monomers per type
        self.c0_stock_single_oligo = c0_stock_single_oligo
        self.c0_stock_all_oligos = c0_stock_all_oligos
        # chacteristic length scale of exponential decay in oligomer length
        # away from monomers
        self.Lambda_stock = Lambda_stock

        if (not self.c0_stock_single_oligo is None) and (not self.c0_vcg_single_oligo is None) \
           and (self.c0_stock_all_oligos is None) and (self.c0_vcg_all_oligos is None):
            self.c0_input_format = 'c0_single_oligo'
        elif (self.c0_stock_single_oligo is None) and (self.c0_vcg_single_oligo is None) \
           and (not self.c0_stock_all_oligos is None) and (not self.c0_vcg_all_oligos is None):
            self.c0_input_format = 'c0_all_oligos'
        elif (self.c0_stock_single_oligo is None) and (self.c0_vcg_single_oligo is None) \
            and (not self.c0_stock_all_oligos is None) and (self.c0_vcg_all_oligos is None) \
            and (optional_parameters['continuous_profile']==True):
            self.c0_input_format = 'continuous_profile'
        elif (self.c0_stock_all_oligos is None) and (self.c0_vcg_all_oligos is None) \
            and (self.c0_stock_single_oligo is None) and (self.c0_stock_all_oligos is None) \
            and (not cs_ss_comb_tot is None):
            self.c0_input_format = 'free_input'
            self.cs_ss_comb_tot = cs_ss_comb_tot
        else:
            self.c0_input_format = ''

        # compute total concentrations including combinatorial multiplicities
        if self.c0_input_format == 'c0_single_oligo':
            self.compute_total_concentrations_given_c0_single_oligo()
        elif self.c0_input_format == 'c0_all_oligos':
            self.compute_total_concentrations_given_c0_all_oligos()
        elif self.c0_input_format == 'continuous_profile':
            self.compute_total_concentrations_continuous_profile()
        elif self.c0_input_format == 'free_input':
            self.compute_total_concentration_given_cs_ss_comb_tot()
        self.is_converged = False
        self.is_converged_log = False


    def compute_total_concentration_given_cs_ss_comb_tot(self):

        self.cs_ss_tot = [] # total concentration for all strands of given length per type
        self.c0_stock_all_oligos = 0.
        self.c0_vcg_all_oligos = 0.

        for iL, L in enumerate(self.cmplxs.Ls):

            if L < self.cmplxs.l_unique:
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                self.cs_ss_tot.append(self.cs_ss_comb_tot[iL]/comb)
                self.c0_stock_all_oligos += self.cs_ss_comb_tot[iL]
            
            elif L >= self.cmplxs.l_unique:
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                self.cs_ss_tot.append(self.cs_ss_comb_tot[iL]/comb)
                self.c0_vcg_all_oligos += self.cs_ss_comb_tot[iL]


    def compute_total_concentrations_given_c0_single_oligo(self):

        self.cs_ss_comb_tot = [] # total concentration for all strand of given length
        self.cs_ss_tot = [] # total concentration for all strands of given length per type
        
        weighted_combs_stock = 0
        weighted_combs_vcg = 0

        for L in self.cmplxs.Ls:

            if L < self.cmplxs.l_unique:
                w_rel = np.exp(-(L-1)/self.Lambda_stock)
                c = self.c0_stock_single_oligo*w_rel
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                weighted_combs_stock += comb*w_rel
                self.cs_ss_comb_tot.append(c*comb)
                self.cs_ss_tot.append(c)
            
            elif L >= self.cmplxs.l_unique:
                w_rel = np.exp(- np.abs(L-self.cmplxs.L_vcg)/self.Lambda_vcg)
                c = self.c0_vcg_single_oligo*w_rel
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                weighted_combs_vcg += comb*w_rel
                self.cs_ss_comb_tot.append(c*comb)
                self.cs_ss_tot.append(c)

        self.cs_ss_comb_tot = np.asarray(self.cs_ss_comb_tot)
        self.cs_ss_tot = np.asarray(self.cs_ss_tot)

        self.c0_stock_all_oligos = self.c0_stock_single_oligo * weighted_combs_stock
        self.c0_vcg_all_oligos = self.c0_vcg_single_oligo * weighted_combs_vcg
    
    
    def compute_total_concentrations_given_c0_all_oligos(self):

        self.cs_ss_comb_tot = [] # total concentration for all strand of given length
        self.cs_ss_tot = [] # total concentration for all strands of given length per type
        
        weigths_stock = 0
        weigths_vcg = 0
        
        for L in self.cmplxs.Ls:

            if L < self.cmplxs.l_unique:
                weigths_stock += np.exp(-(L-1)/self.Lambda_stock)
                
            elif L >= self.cmplxs.l_unique:
                weigths_vcg += np.exp(- np.abs(L-self.cmplxs.L_vcg)/self.Lambda_vcg)
                
        c0_stock_comb_tot = self.c0_stock_all_oligos/weigths_stock
        c0_vcg_comb_tot = self.c0_vcg_all_oligos/weigths_vcg

        for L in self.cmplxs.Ls:

            if L < self.cmplxs.l_unique:
                c = c0_stock_comb_tot*np.exp(-(L-1)/self.Lambda_stock)
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                self.cs_ss_comb_tot.append(c)
                self.cs_ss_tot.append(c/comb)
                
            elif L >= self.cmplxs.l_unique:
                c = c0_vcg_comb_tot*np.exp(- np.abs(L-self.cmplxs.L_vcg)/self.Lambda_vcg)
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                self.cs_ss_comb_tot.append(c)
                self.cs_ss_tot.append(c/comb)

        self.cs_ss_comb_tot = np.asarray(self.cs_ss_comb_tot)
        self.cs_ss_tot = np.asarray(self.cs_ss_tot)


    def compute_total_concentrations_continuous_profile(self):

        # check requirements
        assert self.cmplxs.l_unique == self.cmplxs.L_vcg_min
        assert np.all(self.cmplxs.Ls == np.arange(1, self.cmplxs.L_vcg_max+1, 1))

        self.cs_ss_comb_tot = [] # total concentration for all strand of given length
        self.cs_ss_tot = [] # total concentration for all strands of given length per type

        self.c0_vcg_all_oligos = 0.

        weigths_stock = 0
        for L in self.cmplxs.Ls:
            if L < self.cmplxs.l_unique:
                weigths_stock += np.exp(-(L-1)/self.Lambda_stock)
        c0_stock_comb_tot = self.c0_stock_all_oligos/weigths_stock
        print("c0_stock_comb_tot: ", c0_stock_comb_tot)
        
        for L in range(1, self.cmplxs.l_unique):
            c = c0_stock_comb_tot*np.exp(-(L-1)/self.Lambda_stock)
            comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
            self.cs_ss_comb_tot.append(c)
            self.cs_ss_tot.append(c/comb)
        
        c0_vcg_comb_tot = c0_stock_comb_tot*np.exp(-(self.cmplxs.L_vcg_min-1)/self.Lambda_stock)
        for L in range(self.cmplxs.L_vcg_min, self.cmplxs.L_vcg_max+1, 1):
            c = c0_vcg_comb_tot*np.exp(-(L-self.cmplxs.L_vcg_min)/self.Lambda_vcg)
            comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
            self.cs_ss_comb_tot.append(c)
            self.cs_ss_tot.append(c/comb)
            self.c0_vcg_all_oligos += c
        
        self.cs_ss_comb_tot = np.asarray(self.cs_ss_comb_tot)
        self.cs_ss_tot = np.asarray(self.cs_ss_tot)


    def compute_ratio_concentration_strand_to_reference_strand(self):

        # computes the ratio of the concentration of a strand to the concentration
        # of its reference strand, e. g. for a strand of length 
        # L \in [L_vcg - dL_Vcg, L_vcg + dL_vcg] it returns c(L)/c(L_avg)

        self.crs_ss_tot = np.zeros(len(self.cmplxs.Ls))

        for i, L in enumerate(self.cmplxs.Ls):

            if L < self.cmplxs.l_unique:
                comb_rel = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]] \
                           / self.cmplxs.combs[self.cmplxs.key2indexcmplx[((1,),())]]
                self.crs_ss_tot[i] = np.exp(-(L-1)/self.Lambda_stock)/comb_rel
            
            elif L >= self.cmplxs.l_unique:
                comb_rel = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]] \
                           / self.cmplxs.combs[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]] 
                self.crs_ss_tot[i] = np.exp(- np.abs(L-self.cmplxs.L_vcg)/self.Lambda_vcg)/comb_rel

    def set_ratio_concentration_strand_to_reference_strand_to_one(self):
        self.crs_ss_tot = np.ones(len(self.cmplxs.Ls))
    
    def compute_complex_concentrations_log(self, cs_ss_log):

        cs_cmplxs = compute_complex_concentrations_log_njit(\
                                cs_ss_log, len(self.cmplxs.key2indexcmplx), \
                                self.cmplxs.indexcmplx2indicesstrands_arr, \
                                self.cmplxs.combs_log, self.cmplxs.Kds_log)
        
        return cs_cmplxs
    
    
    def compute_complex_weights(self):

        # computes the weight of each complex including the combinatorial prefactors, 
        # the dissociation constant as well as the concentration of each strand
        # relative to the respective concentration of the reference strand

        self.ws_cmplxs = np.zeros(len(self.cmplxs.key2indexcmplx))

        for i_cmplx, is_strands in self.cmplxs.indexcmplx2indicesstrands.items():

            self.ws_cmplxs[i_cmplx] = \
                self.cmplxs.combs[i_cmplx]*np.prod(self.crs_ss_tot[is_strands])/self.cmplxs.Kds[i_cmplx]

    
    def compute_mass_conservation_log(self, cs_ss_log):
        
        diff = compute_mass_conservation_log_njit(cs_ss_log, \
                                len(self.cmplxs.key2indexcmplx), \
                                self.cmplxs.indexcmplx2indicesstrands_arr, \
                                self.cmplxs.indexstrand2indicescmplxs_arr, \
                                self.cmplxs.combs_log, self.cmplxs.Kds_log, \
                                self.cs_ss_comb_tot)

        return diff
    
    
    def compute_initial_guess_for_computation_of_equilibrium_concentration(\
            self):
        
        include_triplexes=True
        if self.c0_input_format != 'free_input':
            cmplxs = ComplexConstructor(l_unique=self.cmplxs.l_unique, \
                                        alphabet=self.cmplxs.alphabet, \
                                        L_vcg=self.cmplxs.L_vcg, \
                                        L_vcg_min=self.cmplxs.L_vcg_min, \
                                        L_vcg_max=self.cmplxs.L_vcg_max, \
                                        Lmax_stock=self.cmplxs.Lmax_stock, \
                                        comb_vcg=self.cmplxs.comb_vcg, \
                                        gamma_2m=self.cmplxs.gamma_2m, \
                                        gamma_d=self.cmplxs.gamma_d, \
                                        include_triplexes=include_triplexes, \
                                        include_tetraplexes=False)
            cncs = ConcentrationComputer(cmplxs=cmplxs, \
                                         c0_vcg_all_oligos=self.c0_vcg_all_oligos, \
                                         Lambda_vcg=self.Lambda_vcg, \
                                         c0_stock_all_oligos=self.c0_stock_all_oligos, \
                                         Lambda_stock=self.Lambda_stock)

        elif self.c0_input_format == 'free_input':
            cmplxs = ComplexConstructor(l_unique=self.cmplxs.l_unique, \
                                        alphabet=self.cmplxs.alphabet, \
                                        L_vcg=self.cmplxs.L_vcg, \
                                        L_vcg_min=None, L_vcg_max=None, \
                                        Lmax_stock=self.cmplxs.Lmax_stock, \
                                        Ls=self.cmplxs.Ls, \
                                        comb_vcg=self.cmplxs.comb_vcg, \
                                        gamma_2m=self.cmplxs.gamma_2m, \
                                        gamma_d=self.cmplxs.gamma_d, \
                                        include_triplexes=include_triplexes, \
                                        include_tetraplexes=False)
            cncs = ConcentrationComputer(cmplxs=cmplxs, \
                                         cs_ss_comb_tot=self.cs_ss_comb_tot)
        out = root(cncs.compute_mass_conservation_log, np.log(cncs.cs_ss_tot), \
                   method='lm', tol=1e-30)
        # out = root(cncs.compute_mass_conservation_log, np.log(cncs.cs_ss_tot), \
        #            method='hybr', tol=1e-30)    
        
        return np.exp(out.x)
    

    def compute_equilibrium_concentration_log(self, VERBOSE=False):

        if VERBOSE:
            print("determine initial guess for equilibrium concentration of free strands "\
                  +"by solving duplex hybridization equilibrium")
        cs_ss_init = \
            self.compute_initial_guess_for_computation_of_equilibrium_concentration()
        
        if VERBOSE:
            print("determine equilibrium concentration of free strands "\
                  +"by solving full hybridization equilibrium")
        out = root(self.compute_mass_conservation_log, np.log(cs_ss_init), \
                   method='krylov', \
                   options={'fatol':5e-15*np.sum(self.cs_ss_comb_tot), 'disp':VERBOSE})
    
        if out.success:
            self.is_converged_log = True
            self.cs_ss_equ = np.exp(out.x)
            self.cs_cmplxs_equ = self.compute_complex_concentrations_log(np.log(self.cs_ss_equ))
        
        elif not out.success:
            raise ValueError('equilibrium concentration could not be determined')


    def compute_concentrations_added_nucleotides_productive_cvflvs(self):

        self.cs_nuc_cvflvs = {}

        for _, rt_index in self.cmplxs.react_type_cvflvs_2_index.items():
            
            indices = self.cmplxs.react_type_cvflvs_index_2_cmplx_indices[rt_index]

            cs = np.sum(\
                self.cmplxs.react_type_cvflvs_index_2_cmplx_weights[rt_index] \
                * self.cmplxs.react_type_cvflvs_index_2_added_nucleotides[rt_index]
                * self.cs_cmplxs_equ[indices])
            
            rt_simplified_notation = self.cmplxs.index_2_react_type_cvflvs_simpnot[rt_index]
            self.cs_nuc_cvflvs[rt_simplified_notation] = cs
    
    
    def compute_concentrations_productive_cvflvs(self):

        self.cs_cvflvs = {}

        for _, rt_index in self.cmplxs.react_type_cvflvs_2_index.items():
            
            indices = self.cmplxs.react_type_cvflvs_index_2_cmplx_indices[rt_index]

            cs = np.sum(\
                self.cmplxs.react_type_cvflvs_index_2_cmplx_weights[rt_index] \
                * self.cs_cmplxs_equ[indices])
            
            rt_simplified_notation = self.cmplxs.index_2_react_type_cvflvs_simpnot[rt_index]
            self.cs_cvflvs[rt_simplified_notation] = cs

    
    def compute_concentrations_added_nucleotides_productive_cvfol(self):

        self.cs_nuc_cvfol = {}

        for rt, rt_index in self.cmplxs.react_type_cvfol_2_index.items():

            indices = self.cmplxs.react_type_cvfol_index_2_cmplx_indices[rt_index]

            cs = np.sum(\
                self.cmplxs.react_type_cvfol_index_2_cmplx_weights[rt_index] \
                * self.cmplxs.react_type_cvfol_index_2_added_nucleotides[rt_index] \
                * self.cs_cmplxs_equ[indices])

            self.cs_nuc_cvfol[rt] = cs

    
    def compute_concentrations_productive_cvfolp(self):

        self.cs_cvfolp = {}

        for rt, rt_index in self.cmplxs.react_type_cvfolp_2_index.items():

            indices = self.cmplxs.react_type_cvfolp_index_2_cmplx_indices[rt_index]

            cs = np.sum(\
                self.cmplxs.react_type_cvfolp_index_2_cmplx_weights[rt_index] \
                * self.cs_cmplxs_equ[indices])
        
            self.cs_cvfolp[rt] = cs

    
    def compute_weights_productive_cvfolp(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_weights_productive_cvfolp not well defined for tetraplexes') 
        
        self.ws_cvfolp = {}

        for rt, rt_index in self.cmplxs.react_type_cvfolp_2_index.items():

            indices = self.cmplxs.react_type_cvfolp_index_2_cmplx_indices[rt_index]

            ws = np.sum(\
                self.cmplxs.react_type_cvfolp_index_2_cmplx_weights[rt_index] \
                * self.ws_cmplxs[indices])
        
            self.ws_cvfolp[rt] = ws
    
    
    def compute_concentrations_added_nucleotides_productive_cvfolp(self):

        self.cs_nuc_cvfolp = {}

        for rt, rt_index in self.cmplxs.react_type_cvfolp_2_index.items():

            indices = self.cmplxs.react_type_cvfolp_index_2_cmplx_indices[rt_index]

            cs = np.sum(\
                self.cmplxs.react_type_cvfolp_index_2_cmplx_weights[rt_index] \
                * self.cmplxs.react_type_cvfolp_index_2_added_nucleotides[rt_index] \
                * self.cs_cmplxs_equ[indices])
        
            self.cs_nuc_cvfolp[rt] = cs
    
    def compute_concentrations_consumed_oligomers_via_extension_by_monomer(self):        
        
        self.cs_consumed_oligos_by_monomer = {}
        for rt in self.cmplxs.react_type_cvfolp_2_index.keys():
            _, L1, L2 = eval(rt.split('_')[0])
            if L1 == 1 and L2 >= self.cmplxs.l_unique:
                if not L2 in self.cs_consumed_oligos_by_monomer:
                    self.cs_consumed_oligos_by_monomer[L2] \
                        = self.cs_cvfolp[rt]
                else:
                    self.cs_consumed_oligos_by_monomer[L2] \
                        += self.cs_cvfolp[rt]
            elif L1 >= self.cmplxs.l_unique and L2 == 1:
                if not L1 in self.cs_consumed_oligos_by_monomer:
                    self.cs_consumed_oligos_by_monomer[L1] \
                        = self.cs_cvfolp[rt]
                else:
                    self.cs_consumed_oligos_by_monomer[L1] \
                        += self.cs_cvfolp[rt]

    
    def compute_concentrations_productive_co(self):

        self.cs_co = np.zeros(len(self.cmplxs.Ls))

        for _, rt_index in self.cmplxs.react_type_co_2_index.items():
            indices_cmplxs = self.cmplxs.react_type_co_2_cmplx_indices[rt_index]
            cs = np.sum(self.cs_cmplxs_equ[indices_cmplxs])
            self.cs_co[rt_index] = cs
    

    def compute_concentrations_productive_co_fs(self):

        self.cs_co_fs = np.zeros(len(self.cmplxs.react_type_co_fs_2_index))

        for _, rt_index in self.cmplxs.react_type_co_fs_2_index.items():
            indices_cmplxs = self.cmplxs.react_type_co_fs_2_cmplx_indices[rt_index]
            cs = np.sum(self.cs_cmplxs_equ[indices_cmplxs])
            self.cs_co_fs[rt_index] = cs


    def compute_concentrations_productive_hl(self):

        self.cs_hl = np.zeros(len(self.cmplxs.react_type_hl_2_index))

        for _, rt_index in self.cmplxs.react_type_hl_2_index.items():
            indices_cmplxs = self.cmplxs.react_type_hl_2_cmplx_indices[rt_index]
            cs = np.sum(self.cs_cmplxs_equ[indices_cmplxs])
            self.cs_hl[rt_index] = cs

    
    def compute_concentrations_productive_an(self):

        self.cs_an = np.zeros(len(self.cmplxs.react_type_an_2_index))

        for _, rt_index in self.cmplxs.react_type_an_2_index.items():
            indices_cmplxs = self.cmplxs.react_type_an_2_cmplx_indices[rt_index]
            cs = np.sum(self.cs_cmplxs_equ[indices_cmplxs])
            self.cs_an[rt_index] = cs


    def compute_yield_ratio_cvflvs_exact(self):

        if not hasattr(self, 'cs_nuc_cvflvs'):
            self.compute_concentrations_added_nucleotides_productive_cvflvs()
        
        num = 0.
        denom = 0.

        for rt in self.cs_nuc_cvflvs.keys():
            denom += self.cs_nuc_cvflvs[rt]
            if rt.split('_')[2] == 'v': # the produced strand belongs to the "VCG stock"
                num += self.cs_nuc_cvflvs[rt]
        
        self.yield_ratio_cvflvs_exact = num/denom

    
    def compute_yield_ratio_monomer_addition_exact(self):

        if not hasattr(self, 'cs_nuc_cvfolp'):
            self.compute_concentrations_added_nucleotides_productive_cvfolp()

        num = 0.
        denom = 0.

        for rt, c in self.cs_nuc_cvfolp.items():
            _, Leduct1, Leduct2 = eval(rt.split('_')[0])
            if ((Leduct1 == 1) or (Leduct2 == 1)):
                if (Leduct1 + Leduct2 >= self.cmplxs.l_unique):
                    num += c
                denom += c

        self.yield_ratio_monomer_addition_exact = num/denom


    def compute_error_ratio_cvflvs_exact(self):

        if not hasattr(self, 'cs_nuc_cvflvs'):
            self.compute_concentrations_added_nucleotides_productive_cvflvs()
        
        num = 0.
        denom = 0.

        for rt in self.cs_nuc_cvflvs.keys():
            denom += self.cs_nuc_cvflvs[rt]
            if rt.split('_')[3] == 'f': # the produced strand has a wrong sequence
                num += self.cs_nuc_cvflvs[rt]
        
        self.error_ratio_cvflvs_exact = num/denom

    
    def compute_ratio_subtypes_exact(self, subtypes):

        if not hasattr(self, 'cs_nuc_cvflvs'):
            self.compute_concentrations_added_nucleotides_productive_cvflvs()
        
        if not hasattr(self, 'ratio_cvflvs_subtypes_exact'):
            self.ratio_cvflvs_subtypes_exact = {}

        num = 0.
        denom = 0.

        for rt in self.cs_nuc_cvflvs.keys():
            denom += self.cs_nuc_cvflvs[rt]
            if rt in subtypes:
                num += self.cs_nuc_cvflvs[rt]
        
        self.ratio_cvflvs_subtypes_exact[tuple(subtypes)] = num/denom
        

    def compute_error_ratio_cvflvs_fv_v_v_f_only_exact(self):

        if not hasattr(self, 'cs_nuc_cvflvs'):
            self.compute_concentrations_added_nucleotides_productive_cvflvs()
        
        num = self.cs_nuc_cvflvs['fv_v_v_f']
        denom = np.sum(np.asarray(list(self.cs_nuc_cvflvs.values())))

        self.error_ratio_cvflvs_fv_v_v_f_only_exact = num/denom

    
    def compute_errorfree_ratio_cvflvs_exact(self):

        if not hasattr(self, 'cs_nuc_cvflvs'):
            self.compute_concentrations_added_nucleotides_productive_cvflvs()
        
        num = 0.
        denom = 0.

        for rt in self.cs_nuc_cvflvs.keys():
            denom += self.cs_nuc_cvflvs[rt]
            if (rt.split('_')[2] == 'v') and (rt.split('_')[3] == 'c'): 
                # the produced strand belongs to the VCG stock and has correct sequence
                num += self.cs_nuc_cvflvs[rt]
        
        self.errorfree_ratio_cvflvs_exact = num/denom


    def compute_errorfree_ratio_cvflvs_ff_v_v_c_missing_exact(self):

        if not hasattr(self, 'cs_nuc_cvflvs'):
            self.compute_concentrations_added_nucleotides_productive_cvflvs()
        
        num = 0.
        denom = 0.

        for rt in self.cs_nuc_cvflvs.keys():
            denom += self.cs_nuc_cvflvs[rt]
            if (rt.split('_')[2] == 'v') and (rt.split('_')[3] == 'c') and (rt != 'ff_v_v_c'):
                # the produced strand belongs to the VCG stock and has correct sequence
                # but ff_v_v_c is excluded
                num += self.cs_nuc_cvflvs[rt]
        
        self.errorfree_ratio_cvflvs_ff_v_v_c_missing_exact = num/denom

    
    def compute_errorfree_ratio_cvflvs_ff_v_v_c_only_exact(self):

        if not hasattr(self, 'cs_nuc_cvflvs'):
            self.compute_concentrations_added_nucleotides_productive_cvflvs()
        
        num = self.cs_nuc_cvflvs['ff_v_v_c']
        denom = np.sum(np.asarray(list(self.cs_nuc_cvflvs.values())))
        
        self.errorfree_ratio_cvflvs_ff_v_v_c_only_exact = num/denom

    def compute_errorfree_ratio_monomer_addition_exact(self):

        if not hasattr(self, 'cs_nuc_cvfolp'):
            self.compute_concentrations_added_nucleotides_productive_cvfolp()

        num = 0.
        denom = 0.

        for rt, c in self.cs_nuc_cvfolp.items():
            Ls, gen_comp = rt.split('_')
            _, Leduct1, Leduct2 = eval(Ls)
            if ((Leduct1 == 1) or (Leduct2 == 1)):
                # monomer addition
                if (Leduct1 + Leduct2 >= self.cmplxs.l_unique) and (gen_comp == 'c'):
                    num += c
                denom += c
        
        self.errorfree_ratio_monomer_addition_exact = num/denom
    

    def compute_fidelity_monomer_addition_exact(self):

        if not hasattr(self, 'cs_nuc_cvfolp'):
            self.compute_concentrations_added_nucleotides_productive_cvfolp()
        
        num = 0.
        denom = 0.

        for rt, c in self.cs_nuc_cvfolp.items():
            Ls, gen_comp = rt.split('_')
            _, Leduct1, Leduct2 = eval(Ls)
            if ((Leduct1 == 1) or (Leduct2 == 1)) and (Leduct1+Leduct2 >= self.cmplxs.l_unique):
                if gen_comp == 'c':
                    num += c
                denom += c
        
        self.fidelity_monomer_addition_exact = num/denom

    
    def compute_weights_productive_cvflvs(self):

        self.ws_cvflvs = {}

        for rt, rt_index in self.cmplxs.react_type_cvflvs_2_index.items():
            
            indices = self.cmplxs.react_type_cvflvs_index_2_cmplx_indices[\
                self.cmplxs.react_type_cvflvs_2_index[rt]]
            
            ws = np.sum(\
                    self.cmplxs.react_type_cvflvs_index_2_cmplx_weights[rt_index] \
                    * self.ws_cmplxs[indices])

            rt_simplified_notation = self.cmplxs.index_2_react_type_cvflvs_simpnot[rt_index]
            self.ws_cvflvs[rt_simplified_notation] = ws
    

    def compute_weights_added_nucleotides_productive_cvflvs(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_weights_added_nucleotides_productive_cvflvs not well defined for tetraplexes') 

        self.ws_nuc_cvflvs = {}

        for rt, rt_index in self.cmplxs.react_type_cvflvs_2_index.items():
            
            indices = self.cmplxs.react_type_cvflvs_index_2_cmplx_indices[\
                self.cmplxs.react_type_cvflvs_2_index[rt]]
            
            ws = np.sum(\
                self.cmplxs.react_type_cvflvs_index_2_cmplx_weights[rt_index] \
                * self.cmplxs.react_type_cvflvs_index_2_added_nucleotides[rt_index] \
                * self.ws_cmplxs[indices])

            rt_simplified_notation = self.cmplxs.index_2_react_type_cvflvs_simpnot[rt_index]
            self.ws_nuc_cvflvs[rt_simplified_notation] = ws
    

    def compute_weights_added_nucleotides_productive_cvfol(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_weights_added_nucleotides_productive_cvfol not well defined for tetraplexes')
        
        self.ws_nuc_cvfol = {}

        for rt, rt_index in self.cmplxs.react_type_cvfol_2_index.items():

            indices = self.cmplxs.react_type_cvfol_index_2_cmplx_indices[\
                self.cmplxs.react_type_cvfol_2_index[rt]]
            
            ws = np.sum(\
                self.cmplxs.react_type_cvfol_index_2_cmplx_weights[rt_index] \
                * self.cmplxs.react_type_cvfol_index_2_added_nucleotides[rt_index] \
                * self.ws_cmplxs[indices])

            self.ws_nuc_cvfol[rt] = ws
            

    def compute_coefficients_all_cvflvs(self):
        
        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_coefficients_all_cvflvs not well defined for tetraplexes') 
        
        self.alpha_all = self.ws_nuc_cvflvs['ff_f_f_c'] + self.ws_nuc_cvflvs['ff_f_v_c'] \
                         + self.ws_nuc_cvflvs['ff_f_v_f']
        self.beta_all = self.ws_nuc_cvflvs['fv_f_v_c'] + self.ws_nuc_cvflvs['fv_f_v_f'] \
                        + self.ws_nuc_cvflvs['ff_v_f_c'] + \
                        + self.ws_nuc_cvflvs['ff_v_v_c'] + self.ws_nuc_cvflvs['ff_v_v_f']
        self.gamma_all = self.ws_nuc_cvflvs['fv_v_v_c'] + self.ws_nuc_cvflvs['fv_v_v_f'] \
                         + self.ws_nuc_cvflvs['vv_f_v_c'] + self.ws_nuc_cvflvs['vv_f_v_f']
        self.delta_all = self.ws_nuc_cvflvs['vv_v_v_c'] + self.ws_nuc_cvflvs['vv_v_v_f']

    def compute_coefficients_all_cvflvs_nonuc(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_coefficients_all_cvflvs_nonuc not well defined for tetraplex')
        
        self.alpha_all_nonuc = self.ws_cvflvs['ff_f_f_c'] + self.ws_cvflvs['ff_f_v_c'] \
                               + self.ws_cvflvs['ff_f_v_f']
        self.beta_all_nonuc = self.ws_cvflvs['fv_f_v_c'] + self.ws_cvflvs['fv_f_v_f'] \
                        + self.ws_cvflvs['ff_v_f_c'] + \
                        + self.ws_cvflvs['ff_v_v_c'] + self.ws_cvflvs['ff_v_v_f']
        self.gamma_all_nonuc = self.ws_cvflvs['fv_v_v_c'] + self.ws_cvflvs['fv_v_v_f'] \
                         + self.ws_cvflvs['vv_f_v_c'] + self.ws_cvflvs['vv_f_v_f']
        self.delta_all_nonuc = self.ws_cvflvs['vv_v_v_c'] + self.ws_cvflvs['vv_v_v_f']

    def compute_coefficients_yield_cvflvs(self):
        
        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_coefficients_yield_cvflvs not well defined for tetraplexes') 
        
        self.alpha_yield = self.ws_nuc_cvflvs['ff_f_v_c'] + self.ws_nuc_cvflvs['ff_f_v_f']
        self.beta_yield = self.ws_nuc_cvflvs['fv_f_v_c'] + self.ws_nuc_cvflvs['fv_f_v_f'] \
                     + self.ws_nuc_cvflvs['ff_v_v_c'] + self.ws_nuc_cvflvs['ff_v_v_f']
        self.gamma_yield = self.ws_nuc_cvflvs['fv_v_v_c'] + self.ws_nuc_cvflvs['fv_v_v_f'] \
                     + self.ws_nuc_cvflvs['vv_f_v_c'] + self.ws_nuc_cvflvs['vv_f_v_f']
        self.delta_yield = self.ws_nuc_cvflvs['vv_v_v_c'] + self.ws_nuc_cvflvs['vv_v_v_f']
    
    def compute_coefficients_yield_cvflvs_nonuc(self):
        
        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_coefficients_yield_cvflvs not well defined for tetraplexes') 
        
        self.alpha_yield_nonuc = self.ws_cvflvs['ff_f_v_c'] + self.ws_cvflvs['ff_f_v_f']
        self.beta_yield_nonuc = self.ws_cvflvs['fv_f_v_c'] + self.ws_cvflvs['fv_f_v_f'] \
                     + self.ws_cvflvs['ff_v_v_c'] + self.ws_cvflvs['ff_v_v_f']
        self.gamma_yield_nonuc = self.ws_cvflvs['fv_v_v_c'] + self.ws_cvflvs['fv_v_v_f'] \
                     + self.ws_cvflvs['vv_f_v_c'] + self.ws_cvflvs['vv_f_v_f']
        self.delta_yield_nonuc = self.ws_cvflvs['vv_v_v_c'] + self.ws_cvflvs['vv_v_v_f']
    
    def compute_coefficients_error_cvflvs(self):
        
        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_coefficients_error_cvflvs not well defined for tetraplexes') 
        
        self.alpha_err = self.ws_nuc_cvflvs['ff_f_v_f']
        self.beta_err = self.ws_nuc_cvflvs['fv_f_v_f'] + self.ws_nuc_cvflvs['ff_v_v_f']
        self.gamma_err = self.ws_nuc_cvflvs['fv_v_v_f'] + self.ws_nuc_cvflvs['vv_f_v_f']
        self.delta_err = self.ws_nuc_cvflvs['vv_v_v_f']
    
    def compute_coefficients_error_cvflvs_nonuc(self):
        
        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_coefficients_error_cvflvs not well defined for tetraplexes') 
        
        self.alpha_err_nonuc = self.ws_cvflvs['ff_f_v_f']
        self.beta_err_nonuc = self.ws_cvflvs['fv_f_v_f'] + self.ws_cvflvs['ff_v_v_f']
        self.gamma_err_nonuc = self.ws_cvflvs['fv_v_v_f'] + self.ws_cvflvs['vv_f_v_f']
        self.delta_err_nonuc = self.ws_cvflvs['vv_v_v_f']

    def compute_coefficients_errorfree_cvflvs(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_coefficients_errorfree_cvflvs not well defined for tetraplexes') 

        self.alpha_corr = self.ws_nuc_cvflvs['ff_f_v_c']
        self.beta_corr = self.ws_nuc_cvflvs['fv_f_v_c'] + self.ws_nuc_cvflvs['ff_v_v_c']
        self.gamma_corr = self.ws_nuc_cvflvs['fv_v_v_c'] + self.ws_nuc_cvflvs['vv_f_v_c']
        self.delta_corr = self.ws_nuc_cvflvs['vv_v_v_c']

    def compute_coefficients_errorfree_cvflvs_nonuc(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_coefficients_errorfree_cvflvs not well defined for tetraplexes') 

        self.alpha_corr_nonuc = self.ws_cvflvs['ff_f_v_c']
        self.beta_corr_nonuc = self.ws_cvflvs['fv_f_v_c'] + self.ws_cvflvs['ff_v_v_c']
        self.gamma_corr_nonuc = self.ws_cvflvs['fv_v_v_c'] + self.ws_cvflvs['vv_f_v_c']
        self.delta_corr_nonuc = self.ws_cvflvs['vv_v_v_c']

    def compute_yield_ratio_cvflvs(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_yield_ratio_cvflvs not well defined for tetraplexes') 

        if not hasattr(self, 'ws_nuc_cvflvs'):
            self.compute_weights_added_nucleotides_productive_cvflvs()
        
        if not hasattr(self, 'self.alpha_yield'):
            self.compute_coefficients_yield_cvflvs()
        
        if not hasattr(self, 'self.alpha_all'):
            self.compute_coefficients_all_cvflvs()

        c_equ_stock = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((1,),())]]
        c_equ_vcg = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]]
        ratio_cvcgequ_cstockequ = c_equ_vcg/c_equ_stock
        
        num = self.alpha_yield + self.beta_yield*ratio_cvcgequ_cstockequ \
              + self.gamma_yield*ratio_cvcgequ_cstockequ**2 \
              + self.delta_yield*ratio_cvcgequ_cstockequ**3
        
        denom = self.alpha_all + self.beta_all*ratio_cvcgequ_cstockequ + \
                + self.gamma_all*ratio_cvcgequ_cstockequ**2 \
                + self.delta_all*ratio_cvcgequ_cstockequ**3

        self.yield_ratio_cvflvs = num/denom
    
    
    def compute_error_ratio_cvflvs(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_error_ratio_cvflvs not well defined for tetraplexes')

        if not hasattr(self, 'ws_nuc_cvflvs'):
            self.compute_weights_added_nucleotides_productive_cvflvs()

        if not hasattr(self, 'self.alpha_err'):
            self.compute_coefficients_error_cvflvs()
        
        if not hasattr(self, 'self.alpha_all'):
            self.compute_coefficients_all_cvflvs()

        c_equ_stock = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((1,),())]]
        c_equ_vcg = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]]
        ratio_cvcgequ_cstockequ = c_equ_vcg/c_equ_stock
        
        num = self.alpha_err + self.beta_err*ratio_cvcgequ_cstockequ \
              + self.gamma_err*ratio_cvcgequ_cstockequ**2 + \
              + self.delta_err*ratio_cvcgequ_cstockequ**3
        denom = self.alpha_all + self.beta_all*ratio_cvcgequ_cstockequ \
              + self.gamma_all*ratio_cvcgequ_cstockequ**2 \
              + self.delta_all*ratio_cvcgequ_cstockequ**3

        self.error_ratio_cvflvs = num/denom

    def compute_error_ratio_cvflvs_fv_v_v_f_only(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_error_ratio_cvflvs_fv_v_v_f_only not well defined for tetraplexes')

        if not hasattr(self, 'ws_nuc_cvflvs'):
            self.compute_weights_added_nucleotides_productive_cvflvs()

        if not hasattr(self, 'self.alpha_err'):
            self.compute_coefficients_error_cvflvs()
        
        if not hasattr(self, 'self.alpha_all'):
            self.compute_coefficients_all_cvflvs()

        c_equ_stock = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((1,),())]]
        c_equ_vcg = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]]
        ratio_cvcgequ_cstockequ = c_equ_vcg/c_equ_stock
        
        num = self.ws_nuc_cvflvs['fv_v_v_f']*ratio_cvcgequ_cstockequ**2
        denom = self.alpha_all + self.beta_all*ratio_cvcgequ_cstockequ \
              + self.gamma_all*ratio_cvcgequ_cstockequ**2 \
              + self.delta_all*ratio_cvcgequ_cstockequ**3

        self.error_ratio_cvflvs_fv_v_v_f_only = num/denom

    
    def compute_errorfree_ratio_cvflvs(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_errorfree_ratio_cvflvs not well defined for tetraplexes')

        if not hasattr(self, 'ws_nuc_cvflvs'):
            self.compute_weights_added_nucleotides_productive_cvflvs()

        if not hasattr(self, 'self.alpha_corr'):
            self.compute_coefficients_errorfree_cvflvs()
        
        if not hasattr(self, 'self.alpha_all'):
            self.compute_coefficients_all_cvflvs()

        c_equ_stock = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((1,),())]]
        c_equ_vcg = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]]
        ratio_cvcgequ_cstockequ = c_equ_vcg/c_equ_stock
        
        num = self.alpha_corr + self.beta_corr*ratio_cvcgequ_cstockequ \
              + self.gamma_corr*ratio_cvcgequ_cstockequ**2 \
              + self.delta_corr*ratio_cvcgequ_cstockequ**3
        denom = self.alpha_all + self.beta_all*ratio_cvcgequ_cstockequ \
              + self.gamma_all*ratio_cvcgequ_cstockequ**2 \
              + self.delta_all*ratio_cvcgequ_cstockequ**3

        self.errorfree_ratio_cvflvs = num/denom
        self.ratio_cvcgequ_cstockequ = ratio_cvcgequ_cstockequ


    def compute_errorfree_ratio_cvflvs_ff_v_v_c_missing(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_errorfree_ratio_cvflvs_ff_v_v_c_missing not well defined for tetraplexes')

        if not hasattr(self, 'ws_nuc_cvflvs'):
            self.compute_weights_added_nucleotides_productive_cvflvs()

        if not hasattr(self, 'self.alpha_corr'):
            self.compute_coefficients_errorfree_cvflvs()
        
        if not hasattr(self, 'self.alpha_all'):
            self.compute_coefficients_all_cvflvs()

        c_equ_stock = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((1,),())]]
        c_equ_vcg = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]]
        ratio_cvcgequ_cstockequ = c_equ_vcg/c_equ_stock
        
        num = self.alpha_corr + self.ws_nuc_cvflvs['fv_f_v_c']*ratio_cvcgequ_cstockequ \
              + self.gamma_corr*ratio_cvcgequ_cstockequ**2 \
              + self.delta_corr*ratio_cvcgequ_cstockequ**3
        denom = self.alpha_all + self.beta_all*ratio_cvcgequ_cstockequ \
              + self.gamma_all*ratio_cvcgequ_cstockequ**2 \
              + self.delta_all*ratio_cvcgequ_cstockequ**3

        self.errorfree_ratio_cvflvs_ff_v_v_c_missing = num/denom

    
    def compute_errorfree_ratio_cvflvs_ff_v_v_c_only(self):

        if self.cmplxs.include_tetraplexes:
            warnings.warn('compute_errorfree_ratio_cvflvs_ff_v_v_c_only not well defined for tetraplexes')

        if not hasattr(self, 'ws_nuc_cvflvs'):
            self.compute_weights_added_nucleotides_productive_cvflvs()

        if not hasattr(self, 'self.alpha_corr'):
            self.compute_coefficients_errorfree_cvflvs()
        
        if not hasattr(self, 'self.alpha_all'):
            self.compute_coefficients_all_cvflvs()

        c_equ_stock = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((1,),())]]
        c_equ_vcg = self.cs_ss_equ[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]]
        ratio_cvcgequ_cstockequ = c_equ_vcg/c_equ_stock
        
        num = self.ws_nuc_cvflvs['ff_v_v_c']*ratio_cvcgequ_cstockequ
        denom = self.alpha_all + self.beta_all*ratio_cvcgequ_cstockequ \
              + self.gamma_all*ratio_cvcgequ_cstockequ**2 \
              + self.delta_all*ratio_cvcgequ_cstockequ**3

        self.errorfree_ratio_cvflvs_fv_f_v_c_only = num/denom


class ConcentrationOptimizerAddedNucsApproximative(ConcentrationComputer):

    def __init__(self, cmplxs: ComplexConstructor, \
                 Lambda_vcg=None, Lambda_stock=None, \
                 c0_stock_single_oligo=None, c0_stock_all_oligos=None):

        # container of all possible complexes
        self.cmplxs = cmplxs
        if self.cmplxs.include_tetraplexes:
            warnings.warn('ConcentrationOptimizerAddedNucsApproximative not well defined for tetraplexes')
        
        # if not hasattr(self.cmplxs, 'indices_cmplxs_productive'):
        #     self.cmplxs.identify_productive_complexes()

        # characteristic length scale of exponential decay in oligomer length 
        # around L_vcg peak
        self.Lambda_vcg = Lambda_vcg 

        # chacteristic length scale of exponential decay in oligomer length
        # away from monomers
        self.Lambda_stock = Lambda_stock

        # total concentration (per single strand) in feeding stock
        self.c0_stock_single_oligo = c0_stock_single_oligo
        
        # total concentration (all strands, all lengths) in feeding stock
        self.c0_stock_all_oligos = c0_stock_all_oligos

        if (not c0_stock_single_oligo is None) and (c0_stock_all_oligos is None):
            self.c0_input_format = 'c0_single_oligo'
        elif (c0_stock_single_oligo is None) and (not c0_stock_all_oligos is None):
            self.c0_input_format = 'c0_all_oligos'
        else:
            self.c0_input_format = 'c0_none'
        
        if self.c0_input_format == 'c0_single_oligo':
            self.compute_c0_stock_all_oligos()
        elif self.c0_input_format == 'c0_all_oligos':
            self.compute_c0_stock_single_oligo()

        # construct ratio of concentration to concentration of reference strand
        super().compute_ratio_concentration_strand_to_reference_strand()
        super().compute_complex_weights()
        super().compute_weights_added_nucleotides_productive_cvflvs()
        super().compute_coefficients_all_cvflvs()
        super().compute_coefficients_errorfree_cvflvs()
    
    
    def compute_c0_stock_single_oligo(self):

        weighted_combs_stock = 0
        
        for L in self.cmplxs.Ls:
            if L < self.cmplxs.l_unique:
                w_rel = np.exp(-(L-1)/self.Lambda_stock)
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                weighted_combs_stock += comb*w_rel
                
        self.c0_stock_single_oligo = self.c0_stock_all_oligos/weighted_combs_stock
    
    
    def compute_c0_stock_all_oligos(self):

        weighted_combs_stock = 0
        
        for L in self.cmplxs.Ls:
            if L < self.cmplxs.l_unique:
                w_rel = np.exp(-(L-1)/self.Lambda_stock)
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                weighted_combs_stock += comb*w_rel
            
        self.c0_stock_all_oligos = self.c0_stock_single_oligo * weighted_combs_stock

    
    def compute_optimal_equilibrium_concentration_ratio_cvflvs(self):

        c0 = self.alpha_all * self.beta_corr - self.alpha_corr * self.beta_all
        c1 = 2 * (self.alpha_all * self.gamma_corr - self.alpha_corr * self.gamma_all)
        c2 = self.beta_all * self.gamma_corr - self.beta_corr * self.gamma_all \
             + 3 * (self.alpha_all * self.delta_corr - self.alpha_corr * self.delta_all)
        c3 = 2 * (self.beta_all * self.delta_corr - self.beta_corr * self.delta_all)
        c4 = self.gamma_all * self.delta_corr - self.gamma_corr * self.delta_all
        
        d0 = c0/c4
        d1 = c1/c4
        d2 = c2/c4
        d3 = c3/c4

        roots = hf.solve_quartic_equation(d3,d2,d1,d0)
        
        roots_valid = []
        for root in roots:
            if np.abs(np.imag(root)) <= 1e-15*np.abs(np.real(root)) and np.real(root) > 0.:
                roots_valid.append(root)
        
        if len(roots_valid) == 0 or len(roots_valid) > 1:
            self.ratio_cvcgequ_cstockequ_opt = np.inf
            self.errorfree_ratio_cvflvs_opt = np.nan
            
        else:
            self.ratio_cvcgequ_cstockequ_opt = np.real(roots_valid[0])
            self.errorfree_ratio_cvflvs_opt = \
                self.compute_errorfree_ratio_cvflvs_childclass(\
                    self.ratio_cvcgequ_cstockequ_opt)


    def compute_errorfree_ratio_cvflvs_childclass(self, \
                                                  ratio_cvcgequ_cstockequ):

        num = self.alpha_corr + self.beta_corr*ratio_cvcgequ_cstockequ \
              + self.gamma_corr*ratio_cvcgequ_cstockequ**2 \
              + self.delta_corr*ratio_cvcgequ_cstockequ**3
        denom = self.alpha_all + self.beta_all*ratio_cvcgequ_cstockequ \
              + self.gamma_all*ratio_cvcgequ_cstockequ**2 \
              + self.delta_all*ratio_cvcgequ_cstockequ**3

        return num/denom


    def compute_difference_desired_and_given_errorfree_ratio(self, ratio_cvcgequ_cstockequ):
        diff = self.errorfree_ratio_cvflvs_cto \
               - self.compute_errorfree_ratio_cvflvs_childclass(ratio_cvcgequ_cstockequ)
        return diff
    
    
    def compute_close_to_optimal_equilibrium_concentration_ratio_cvflvs(self):

        # define close to optimal error-free ratio
        self.errorfree_ratio_cvflvs_cto = 0.99*self.errorfree_ratio_cvflvs_opt

        # lower bound for the equilibrium concentration ratio
        out = root_scalar(self.compute_difference_desired_and_given_errorfree_ratio, \
                          method='brenth', bracket=[1e-15*self.ratio_cvcgequ_cstockequ_opt, \
                                                    self.ratio_cvcgequ_cstockequ_opt], \
                          xtol=1e-20)
        if out.converged:
            self.ratio_cvcgequ_cstockequ_cto_lb = out.root
            errorfree_ratio_check = self.compute_errorfree_ratio_cvflvs_childclass(\
                self.ratio_cvcgequ_cstockequ_cto_lb)
            assert np.abs(self.errorfree_ratio_cvflvs_cto - errorfree_ratio_check) \
                <= 1e-5*self.errorfree_ratio_cvflvs_cto
        else:
            raise ValueError('computation of lower bound of close-to-optimal concentration '\
                             + 'ratio did not converge')
        
        # lower bound for the equilibrium concentration ratio
        out = root_scalar(self.compute_difference_desired_and_given_errorfree_ratio, \
                          method='brenth', bracket=[self.ratio_cvcgequ_cstockequ_opt, \
                                                    1e15*self.ratio_cvcgequ_cstockequ_opt], \
                          xtol=1e-20)
        if out.converged:
            self.ratio_cvcgequ_cstockequ_cto_ub = out.root
            errorfree_ratio_check = self.compute_errorfree_ratio_cvflvs_childclass(\
                self.ratio_cvcgequ_cstockequ_cto_ub)
            assert np.abs(self.errorfree_ratio_cvflvs_cto - errorfree_ratio_check) \
                <= 1e-5*self.errorfree_ratio_cvflvs_cto
        else:
            raise ValueError('computation of upper bound of close-to-optimal concentration '\
                             + 'ratio did not converge')


    def construct_equilibrium_single_strand_concentration(self, \
                                                          c_equ_stock, \
                                                          ratio_cvcgequ_cstockequ):

        cs_ss_equ = np.zeros(len(self.cmplxs.Ls))

        for i, L in enumerate(self.cmplxs.Ls):
            if L < self.cmplxs.l_unique:
                cs_ss_equ[i] = c_equ_stock*np.exp(-(L-1)/self.Lambda_stock)
            
            elif L >= self.cmplxs.l_unique:
                cs_ss_equ[i] = c_equ_stock*ratio_cvcgequ_cstockequ\
                               * np.exp(- np.abs(L-self.cmplxs.L_vcg)/self.Lambda_vcg)
        
        return cs_ss_equ

    
    def compute_total_concentrations(self, cs_ss_equ):

        cs_cmplxs_equ = self.compute_complex_concentrations_log(np.log(cs_ss_equ))

        cs_ss_comb_tot = np.zeros(len(cs_ss_equ))
        for i_strand, is_cmplxs in self.cmplxs.indexstrand2indicescmplxs.items():
            cs_ss_comb_tot[i_strand] = np.sum(cs_cmplxs_equ[is_cmplxs])
    
        return cs_ss_comb_tot
    
    
    def compute_total_feeding_stock_concentration_all_oligos(self, cs_ss_comb_tot):

        c0_stock_all_oligos_now = 0.
        for i, L in enumerate(self.cmplxs.Ls):
            if L < self.cmplxs.l_unique:
                c0_stock_all_oligos_now += cs_ss_comb_tot[i]
        return c0_stock_all_oligos_now

    
    def compute_total_vcg_concentration_all_oligos(self, cs_ss_comb_tot):

        c0_vcg_all_oligos_now = 0.
        for i, L in enumerate(self.cmplxs.Ls):
            if L >= self.cmplxs.l_unique:
                c0_vcg_all_oligos_now += cs_ss_comb_tot[i]
        return c0_vcg_all_oligos_now
    
    
    def compute_difference_desired_and_given_feeding_stock_concentration(self, \
            c_equ_stock, ratio_cvcgequ_cstockequ):
        
        cs_ss_equ = self.construct_equilibrium_single_strand_concentration(\
            c_equ_stock, ratio_cvcgequ_cstockequ)
        cs_ss_comb_tot = self.compute_total_concentrations(cs_ss_equ)
        diff = self.compute_total_feeding_stock_concentration_all_oligos(cs_ss_comb_tot) \
               - self.c0_stock_all_oligos
        return diff
    
    
    def compute_total_concentration_ratio_given_equilibrium_concentration_ratio(self, \
            ratio_cvcgequ_cstockequ):

        out = root_scalar(self.compute_difference_desired_and_given_feeding_stock_concentration, \
                          args=(ratio_cvcgequ_cstockequ), \
                          method='bisect', bracket=[0, self.c0_stock_single_oligo], \
                          xtol=1e-15*self.c0_stock_single_oligo)
        
        if out.converged:
            c_equ_stock = out.root
            cs_ss_equ = self.construct_equilibrium_single_strand_concentration(\
                    c_equ_stock, ratio_cvcgequ_cstockequ)
            c_equ_vcg = cs_ss_equ[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]]
            ratio_cvcgequ_cstockequ_check = c_equ_vcg/c_equ_stock
            assert np.abs(ratio_cvcgequ_cstockequ_check - ratio_cvcgequ_cstockequ) <= 1e-5*ratio_cvcgequ_cstockequ
            cs_ss_comb_tot = self.compute_total_concentrations(cs_ss_equ)
            c0_vcg_all_oligos = self.compute_total_vcg_concentration_all_oligos(\
                cs_ss_comb_tot)
            c0_stock_all_oligos = self.compute_total_feeding_stock_concentration_all_oligos(\
                cs_ss_comb_tot)
            ratio_cvcgtot_cstocktot = c0_vcg_all_oligos/c0_stock_all_oligos
        
        else:
            raise ValueError('computation of total concentration ratio did not converge')
        
        return ratio_cvcgtot_cstocktot

    
    def compute_optimal_total_concentration_ratio_cvflvs(self):

        if not np.isinf(self.ratio_cvcgequ_cstockequ_opt):
            
            self.ratio_cvcgtot_cstocktot_opt = \
                self.compute_total_concentration_ratio_given_equilibrium_concentration_ratio(\
                    self.ratio_cvcgequ_cstockequ_opt)

        else:
            self.ratio_cvcgtot_cstocktot_opt = np.inf

    
    def compute_close_to_optimal_total_concentration_ratio_cvflvs(self):

        self.ratio_cvcgtot_cstocktot_cto_lb = \
            self.compute_total_concentration_ratio_given_equilibrium_concentration_ratio(\
                self.ratio_cvcgequ_cstockequ_cto_lb)
        self.ratio_cvcgtot_cstocktot_cto_ub = \
            self.compute_total_concentration_ratio_given_equilibrium_concentration_ratio(\
                self.ratio_cvcgequ_cstockequ_cto_ub)
        

class ConcentrationOptimizerAddedNucsExact(ConcentrationComputer):

    def __init__(self, cmplxs: ComplexConstructor, \
                 Lambda_vcg=None, Lambda_stock=None, \
                 c0_stock_single_oligo=None, c0_stock_all_oligos=None):

        # container of all possible complexes
        self.cmplxs = cmplxs
        # if not hasattr(self.cmplxs, 'indices_cmplxs_productive'):
        #     self.cmplxs.identify_productive_complexes()

        # characteristic length scale of exponential decay in oligomer length 
        # around L_vcg peak
        self.Lambda_vcg = Lambda_vcg 

        # chacteristic length scale of exponential decay in oligomer length
        # away from monomers
        self.Lambda_stock = Lambda_stock

        # total concentration (per single strand) in feeding stock
        self.c0_stock_single_oligo = c0_stock_single_oligo
        
        # total concentration (all strands, all lengths) in feeding stock
        self.c0_stock_all_oligos = c0_stock_all_oligos

        if (not c0_stock_single_oligo is None) and (c0_stock_all_oligos is None):
            self.c0_input_format = 'c0_single_oligo'
        elif (c0_stock_single_oligo is None) and (not c0_stock_all_oligos is None):
            self.c0_input_format = 'c0_all_oligos'
        else:
            self.c0_input_format = 'c0_none'
        
        if self.c0_input_format == 'c0_single_oligo':
            self.compute_c0_stock_all_oligos()
        elif self.c0_input_format == 'c0_all_oligos':
            self.compute_c0_stock_single_oligo()
            
    
    def compute_c0_stock_single_oligo(self):

        weighted_combs_stock = 0
        
        for L in self.cmplxs.Ls:
            if L < self.cmplxs.l_unique:
                w_rel = np.exp(-(L-1)/self.Lambda_stock)
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                weighted_combs_stock += comb*w_rel
                
        self.c0_stock_single_oligo = self.c0_stock_all_oligos/weighted_combs_stock
    
    
    def compute_c0_stock_all_oligos(self):

        weighted_combs_stock = 0
        
        for L in self.cmplxs.Ls:
            if L < self.cmplxs.l_unique:
                w_rel = np.exp(-(L-1)/self.Lambda_stock)
                comb = self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]]
                weighted_combs_stock += comb*w_rel
            
        self.c0_stock_all_oligos = self.c0_stock_single_oligo * weighted_combs_stock
        
        
    def compute_errorfree_ratio_cvflvs_exact_childclass(self, ratio_cvcgtot_cstocktot, \
                                                        sign=1.0):
        
        # print("ratio_cvcgtot_cstocktot: ", ratio_cvcgtot_cstocktot)
        if ratio_cvcgtot_cstocktot < 0:
            return 1000
        
        cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                c0_vcg_all_oligos=ratio_cvcgtot_cstocktot*self.c0_stock_all_oligos, \
                Lambda_vcg=self.Lambda_vcg, \
                c0_stock_all_oligos=self.c0_stock_all_oligos, \
                Lambda_stock=self.Lambda_stock)
        cncs.compute_equilibrium_concentration_log(VERBOSE=False)
        cncs.compute_concentrations_added_nucleotides_productive_cvflvs()
        cncs.compute_errorfree_ratio_cvflvs_exact()
    
        return sign*cncs.errorfree_ratio_cvflvs_exact
            
        
    def compute_equilibrium_concentration_ratio(self, ratio_cvcgtot_cstocktot):

        # print("ratio_cvcgtot_cstocktot: ", ratio_cvcgtot_cstocktot)
        cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                c0_vcg_all_oligos=ratio_cvcgtot_cstocktot*self.c0_stock_all_oligos, \
                Lambda_vcg=self.Lambda_vcg, \
                c0_stock_all_oligos=self.c0_stock_all_oligos, \
                Lambda_stock=self.Lambda_stock)
        cncs.compute_equilibrium_concentration_log()
        cncs.compute_concentrations_added_nucleotides_productive_cvflvs()
        cncs.compute_errorfree_ratio_cvflvs_exact()
        
        # old version, without weighting the combinatorics
        # c_equ_stock = cncs.cs_ss_equ[self.cmplxs.key2indexcmplx[((1,),())]]
        # c_equ_vcg = cncs.cs_ss_equ[self.cmplxs.key2indexcmplx[((self.cmplxs.L_vcg,),())]]
        # new version, including weighting by combinatorics
        c_equ_stock = np.sum([self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]] \
                              * cncs.cs_ss_equ[self.cmplxs.key2indexcmplx[((L,)),()]] \
                              for L in self.cmplxs.Ls if L < self.cmplxs.l_unique])
        c_equ_vcg = np.sum([self.cmplxs.combs[self.cmplxs.key2indexcmplx[((L,),())]] \
                              * cncs.cs_ss_equ[self.cmplxs.key2indexcmplx[((L,)),()]] \
                              for L in self.cmplxs.Ls if L >= self.cmplxs.l_unique])
        
        return c_equ_vcg/c_equ_stock
        

    def compute_optimal_total_concentration_ratio_cvflvs_exact(self):

        out = minimize_scalar(self.compute_errorfree_ratio_cvflvs_exact_childclass, \
                              args=(-1.0,), bracket=[1e-9, 1e-1])

        if out.success:
            self.ratio_cvcgtot_cstocktot_opt = out.x
            self.errorfree_ratio_cvflvs_exact_opt = -out.fun
        
        else:
            raise ValueError('computation of the optimal total concentration ratio did not converge')

    
    def compute_optimal_equilibrium_concentration_ratio_cvflvs_exact(self):

        if not hasattr(self, 'ratio_cvcgtot_cstocktot_opt'):
            self.compute_optimal_total_concentration_ratio_cvflvs_exact()

        self.ratio_cvcgequ_cstockequ_opt = \
            self.compute_equilibrium_concentration_ratio(\
                self.ratio_cvcgtot_cstocktot_opt)

    
    def compute_difference_desired_and_given_errorfree_ratio(self, ratio_cvcgtot_cstocktot):
        diff = self.errorfree_ratio_cvflvs_cto \
               - self.compute_errorfree_ratio_cvflvs_exact_childclass(ratio_cvcgtot_cstocktot)
        return diff
    

    def compute_close_to_optimal_total_concentration_ratio_cvflvs(self):

        # define close to optimal error-free ratio
        self.errorfree_ratio_cvflvs_cto = 0.99*self.errorfree_ratio_cvflvs_exact_opt

        # lower bound for the equilibrium concentration ratio
        out = root_scalar(self.compute_difference_desired_and_given_errorfree_ratio, \
                          method='brenth', bracket=[1e-15*self.ratio_cvcgtot_cstocktot_opt, \
                                                    self.ratio_cvcgtot_cstocktot_opt], \
                          xtol=1e-20)
        if out.converged:
            self.ratio_cvcgtot_cstocktot_cto_lb = out.root
            errorfree_ratio_check = self.compute_errorfree_ratio_cvflvs_exact_childclass(\
                self.ratio_cvcgtot_cstocktot_cto_lb)
            assert np.abs(self.errorfree_ratio_cvflvs_cto - errorfree_ratio_check) \
                <= 1e-5*self.errorfree_ratio_cvflvs_cto
        else:
            raise ValueError('computation of lower bound of close-to-optimal concentration '\
                             + 'ratio did not converge')
        
        # upper bound for the equilibrium concentration ratio
        out = root_scalar(self.compute_difference_desired_and_given_errorfree_ratio, \
                          method='brenth', bracket=[self.ratio_cvcgtot_cstocktot_opt, \
                                                    1e3*self.ratio_cvcgtot_cstocktot_opt], \
                          xtol=1e-20)
        if out.converged:
            self.ratio_cvcgtot_cstocktot_cto_ub = out.root
            errorfree_ratio_check = self.compute_errorfree_ratio_cvflvs_exact_childclass(\
                self.ratio_cvcgtot_cstocktot_cto_ub)
            assert np.abs(self.errorfree_ratio_cvflvs_cto - errorfree_ratio_check) \
                <= 1e-5*self.errorfree_ratio_cvflvs_cto
        else:
            raise ValueError('computation of upper bound of close-to-optimal concentration '\
                             + 'ratio did not converge')
        
    
    def compute_close_to_optimal_equilibrium_concentration_ratio_cvflvs(self):

        if not hasattr(self, 'errorfree_ratio_cvflvs_cto'):
            self.compute_close_to_optimal_total_concentration_ratio_cvflvs()
        
        self.ratio_cvcgequ_cstockequ_cto_lb = self.compute_equilibrium_concentration_ratio(\
                                                self.ratio_cvcgtot_cstocktot_cto_lb)
        self.ratio_cvcgequ_cstockequ_cto_ub = self.compute_equilibrium_concentration_ratio(\
                                                self.ratio_cvcgtot_cstocktot_cto_ub)
