#!/bin/env python3

import sys
sys.path.append('../../src/')
import os
from multiprocessing import Pool
import tqdm
from ComplexConstructor import *
from ConcentrationComputer import *
    
class DataPoint__MultiLength:

    def __init__(self, cmplxs, c0_mono_tot, c0_rest_tot):
        self.cmplxs: ComplexConstructor = cmplxs
        self.Lslong = [L for L in self.cmplxs.Ls if L >= self.cmplxs.l_unique]
        self.Llong2index = {L:i for i, L in enumerate(self.Lslong)}
        
        # c0_rest_tot contains total concentration of oligomers between length 2 and 12
        c0_stock = c0_mono_tot + c0_rest_tot/11 # concentration of monomers + dimers
        c0_vcg = 10/11 * c0_rest_tot # concentration of oligomer of length 3 to 12

        self.cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                                    c0_vcg_all_oligos=c0_vcg, Lambda_vcg=np.inf, \
                                    c0_stock_all_oligos=c0_stock, Lambda_stock=1/np.log(c0_mono_tot*10/c0_vcg))
        self.cncs.compute_equilibrium_concentration_log()
        self.cncs.compute_concentrations_productive_cvfolp()
        self.cncs.compute_yield_ratio_monomer_addition_exact()
        self.cncs.compute_fidelity_monomer_addition_exact()
        self.compute_fraction_consumed_oligomers_via_extension_by_monomer()
    
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

class DataSet__MultiLength:

    def __init__(self, L_vcg_min, L_vcg_max, c0_mono_tot, ratios_cresttot_cmonotot, \
                 cmplxs_params=None):
        
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
        
        self.c0_mono_tot = c0_mono_tot
        self.ratios_cresttot_cmonotot = ratios_cresttot_cmonotot
        self.Lslong = [L for L in self.cmplxs.Ls if L >= self.cmplxs.l_unique]
        self.Llong2index = {L:i for i, L in enumerate(self.Lslong)}

    def compute_single_data_point(self, ratio_cresttot_cmonotot):

        dp = DataPoint__MultiLength(cmplxs=self.cmplxs, c0_mono_tot=self.c0_mono_tot, \
                       c0_rest_tot=ratio_cresttot_cmonotot*self.c0_mono_tot)
        dp.compute_fraction_consumed_oligomers_via_extension_by_monomer()
        
        cs_ss_equ = dp.cncs.cs_ss_equ
        cs_ss_comb_equ = dp.cncs.cs_cmplxs_equ[0:len(self.cmplxs.Ls)]
        
        cs_ss_tot = dp.cncs.cs_ss_tot
        cs_ss_comb_tot = dp.cncs.cs_ss_comb_tot
        
        fraction_consumed_oligos_by_monomer = dp.fraction_consumed_oligos_by_monomer

        cs_cvfolp = dp.cncs.cs_cvfolp

        efficiency = dp.cncs.yield_ratio_monomer_addition_exact * dp.cncs.fidelity_monomer_addition_exact

        del dp

        return cs_ss_tot, cs_ss_comb_tot, \
               cs_ss_equ, cs_ss_comb_equ, \
               fraction_consumed_oligos_by_monomer, cs_cvfolp, efficiency

    def compute_all_data_points(self):

        # p = Pool(os.cpu_count()-4)
        # outs = []
        # for out in tqdm.tqdm(p.imap(self.compute_single_data_point, \
        #                             self.ratios_cvcgtot_cstocktot), \
        #                      total=len(self.ratios_cvcgtot_cstocktot)):
        #     outs.append(out)
        # p.close()
        
        outs = []
        for ratio_cvcgtot_cstocktot in tqdm.tqdm(self.ratios_cresttot_cmonotot, \
                                                 total=len(self.ratios_cresttot_cmonotot)):
            out = self.compute_single_data_point(ratio_cvcgtot_cstocktot)
            outs.append(out) 

        self.css_ss_tot = {L:[] for L in self.cmplxs.Ls}
        self.css_ss_comb_tot = {L:[] for L in self.cmplxs.Ls}

        self.css_ss_equ = {L:[] for L in self.cmplxs.Ls}
        self.css_ss_comb_equ = {L:[] for L in self.cmplxs.Ls}

        self.fractions_consumed_oligos_by_monomer = {L:[] for L in self.Lslong}
        self.css_cvfolp =  {key:[] for key in self.cmplxs.react_type_cvfolp_2_index.keys()}

        self.efficiencies = []

        for out in outs:

            cs_ss_tot, cs_ss_comb_tot, \
            cs_ss_equ, cs_ss_comb_equ, \
            fraction_consumed_oligos_by_monomer, cs_cvfolp, efficiency = out

            for iL, L in enumerate(self.cmplxs.Ls):

                self.css_ss_tot[L].append(cs_ss_tot[iL])
                self.css_ss_comb_tot[L].append(cs_ss_comb_tot[iL])
                
                self.css_ss_equ[L].append(cs_ss_equ[iL])
                self.css_ss_comb_equ[L].append(cs_ss_comb_equ[iL])

                if L >= self.cmplxs.l_unique:
                    self.fractions_consumed_oligos_by_monomer[L].append(fraction_consumed_oligos_by_monomer[L])

            for rt in self.cmplxs.react_type_cvfolp_2_index.keys():
                self.css_cvfolp[rt].append(cs_cvfolp[rt])

            self.efficiencies.append(efficiency)
        
        self.css_ss_tot = {key:np.asarray(values) for key, values in self.css_ss_tot.items()}
        self.css_ss_comb_tot = {key:np.asarray(values) \
                                for key, values in self.css_ss_comb_tot.items()}

        self.css_ss_equ = {key:np.asarray(values) for key, values in self.css_ss_equ.items()}
        self.css_ss_comb_equ = {key:np.asarray(values) \
                                for key, values in self.css_ss_comb_equ.items()}
        
        self.fractions_consumed_oligos_by_monomer = \
            {key:np.asarray(values) \
             for key, values in self.fractions_consumed_oligos_by_monomer.items()}
        self.css_cvfolp = {key:np.asarray(values) for key, values in self.css_cvfolp.items()}

        self.efficiencies = np.asarray(self.efficiencies)
