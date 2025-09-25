#!/bin/env python3

import sys
sys.path.append('../../src/')
from multiprocessing import Pool
import os
import tqdm
from typing import List
import pickle as pkl

import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams.update(
    {'text.usetex':False}
)

from ComplexConstructor import *
from ConcentrationComputer import *
import seaborn as sns

class DataPoint:

    def __init__(self, L_vcg_min, L_vcg_max, Lambda_vcg, gamma_2m, c0_stock_all):

        self.L_vcg_min = L_vcg_min
        self.L_vcg_max = L_vcg_max
        self.Lambda_vcg = Lambda_vcg
        self.gamma_2m = gamma_2m
        self.c0_stock_all = c0_stock_all
        self.include_tetraplexes = True

        self.cmplxs = ComplexConstructor(l_unique=3, alphabet=4, \
                    L_vcg=self.L_vcg_min, L_vcg_min=self.L_vcg_min, L_vcg_max=self.L_vcg_max, \
                    Lmax_stock=1, comb_vcg=32, gamma_2m=self.gamma_2m, gamma_d=self.gamma_2m/2, \
                    include_tetraplexes=self.include_tetraplexes)
        self.cmplxs.identify_reaction_types_cvfolp_all_complexes()

        self.cncs = ConcentrationOptimizerAddedNucsExact(cmplxs=self.cmplxs, \
                    Lambda_vcg=self.Lambda_vcg, Lambda_stock=np.inf, \
                    c0_stock_all_oligos=self.c0_stock_all)
        
        # compute optimal ratio cvcgtot to cstocktot
        self.cncs.compute_optimal_total_concentration_ratio_cvflvs_exact()
        self.errorfree_ratio_cvflvs_opt = self.cncs.errorfree_ratio_cvflvs_exact_opt

        # compute equilibrium complex concentration for identified optimal ratio
        self.cncs.c0_vcg_all_oligos = c0_stock_all*self.cncs.ratio_cvcgtot_cstocktot_opt
        self.cncs.compute_total_concentrations_given_c0_all_oligos()
        self.cncs.compute_equilibrium_concentration_log()
        
        # compute cs_added_nucs_cvflvs
        self.cncs.compute_concentrations_added_nucleotides_productive_cvflvs()
        cs_nuc_cvflvs_tot = np.sum(np.asarray(list(self.cncs.cs_nuc_cvflvs.values())))
        self.ratios_nuc_cvflvs = {key:value/cs_nuc_cvflvs_tot \
                                  for key, value in self.cncs.cs_nuc_cvflvs.items()}
        
        # compute cs_added_nucs_cvfolp
        self.cncs.compute_concentrations_added_nucleotides_productive_cvfolp()
        cs_nuc_cvfolp_tot = np.sum(np.asarray(list(self.cncs.cs_nuc_cvfolp.values())))
        self.ratios_nuc_cvfolp = {key:value/cs_nuc_cvfolp_tot \
                                  for key, value in self.cncs.cs_nuc_cvfolp.items()}
        keys_nuc_cvfolp_sorted = sorted(self.ratios_nuc_cvfolp.keys(), \
                                        key = lambda el: self.ratios_nuc_cvfolp[el])
        self.ratios_nuc_cvfolp = {key:self.ratios_nuc_cvfolp[key] for key in keys_nuc_cvfolp_sorted[::-1]}
        

    def identify_relevant_subset_of_ratios_nuc_cvfolp(self, threshold):

        self.ratios_nuc_cvfolp_rel = {}
        rest = 0.
        for key, value in self.ratios_nuc_cvfolp.items():
            if value >= threshold:
                self.ratios_nuc_cvfolp_rel[key] = value
            else:
                rest += value
        self.ratios_nuc_cvfolp_rel['rest'] = rest
    
    def map_cvfolp_to_cvflvs(self):

        self.react_type_cvfolp_2_react_type_cvflvs = {}

        for rt in self.ratios_nuc_cvfolp.keys():
            lt, le1, le2 = eval(rt.split('_')[0])
            cvf = rt.split('_')[1]
            rt_cvflvs = ""
            les, lel = sorted([le1, le2])
            
            if les < self.cmplxs.l_unique:
                rt_cvflvs += "f"
            else:
                rt_cvflvs += "v"
            if lel < self.cmplxs.l_unique:
                rt_cvflvs += "f"
            else:
                rt_cvflvs += "v"
            rt_cvflvs += "_"
            if lt < self.cmplxs.l_unique:
                rt_cvflvs += "f"
            else:
                rt_cvflvs += "v"
            rt_cvflvs += "_"
            if les+lel < self.cmplxs.l_unique:
                rt_cvflvs += "f"
            else:
                rt_cvflvs += "v"
            rt_cvflvs += "_"
            rt_cvflvs += cvf

            self.react_type_cvfolp_2_react_type_cvflvs[rt] = rt_cvflvs


class DataSet:

    def __init__(self, Ls_vcg_min, Ls_vcg_max, Lambdas_vcg, variable, label, \
                 gamma_2m, c0_stock_all):

        if len(Ls_vcg_max) != len(Lambdas_vcg):
            raise ValueError('invalid definition of Ls_vcg_max and Lambda_vcg')
        
        self.Ls_vcg_min = Ls_vcg_min
        self.Ls_vcg_max = Ls_vcg_max
        self.Lambdas_vcg = Lambdas_vcg
        self.variable = variable
        self.label = label
        self.gamma_2m = gamma_2m
        self.c0_stock_all = c0_stock_all
        self.include_tetraplexes = True

        # print("construct reaction types cvflvs and cvfolp")
        print("construct reaction types cvflvs")
        self.construct_reaction_types_cvflvs()
        print("construct data points")
        self.construct_all_data_points()
        print("extract errorfree ratios_cvflvs")
        self.extract_errorfree_ratios_cvflvs()
        print("extract ratios reaction types cvflvs")
        self.extract_ratios_reaction_types_cvflvs_from_data_points()
    
    def construct_reaction_types_cvflvs(self):

        self.reaction_type_cvflvs_2_index = {}
        self.reaction_type_cvfolp_2_index = {}

        if not self.variable == 'kappa':
            for i in range(len(self.Ls_vcg_max)):
                cmplxs = ComplexConstructor(l_unique=3, alphabet=4, \
                        L_vcg=self.Ls_vcg_min[i], L_vcg_min=self.Ls_vcg_min[i], \
                        L_vcg_max=self.Ls_vcg_max[i], Lmax_stock=1, comb_vcg=32, \
                        gamma_2m=self.gamma_2m, gamma_d=self.gamma_2m/2, \
                        include_tetraplexes=self.include_tetraplexes)
                cmplxs.identify_reaction_types_cvfolp_all_complexes()

                for rt in cmplxs.index_2_react_type_cvflvs_simpnot.values():
                    if not rt in self.reaction_type_cvflvs_2_index:
                        if len(self.reaction_type_cvflvs_2_index)!=0:
                            last_key = next(reversed(self.reaction_type_cvflvs_2_index))
                            last_index = self.reaction_type_cvflvs_2_index[last_key]
                            self.reaction_type_cvflvs_2_index[rt] = last_index + 1
                        else:
                            self.reaction_type_cvflvs_2_index[rt] = 0
        
        elif self.variable == 'kappa':
            cmplxs = ComplexConstructor(l_unique=3, alphabet=4, \
                        L_vcg=self.Ls_vcg_min[0], L_vcg_min=self.Ls_vcg_min[0], \
                        L_vcg_max=self.Ls_vcg_max[0], Lmax_stock=1, comb_vcg=32, \
                        gamma_2m=self.gamma_2m, gamma_d=self.gamma_2m/2, \
                    include_tetraplexes=self.include_tetraplexes)
            cmplxs.identify_reaction_types_cvfolp_all_complexes()

            for rt in cmplxs.index_2_react_type_cvflvs_simpnot.values():
                if not rt in self.reaction_type_cvflvs_2_index:
                    if len(self.reaction_type_cvflvs_2_index)!=0:
                        last_key = next(reversed(self.reaction_type_cvflvs_2_index))
                        last_index = self.reaction_type_cvflvs_2_index[last_key]
                        self.reaction_type_cvflvs_2_index[rt] = last_index + 1
                    else:
                        self.reaction_type_cvflvs_2_index[rt] = 0
    

    def construct_reaction_types_cvflvs_and_cvfolp(self):

        self.reaction_type_cvflvs_2_index = {}
        self.reaction_type_cvfolp_2_index = {}

        if not self.variable == 'kappa':
            for i in range(len(self.Ls_vcg_max)):
                cmplxs = ComplexConstructor(l_unique=3, alphabet=4, \
                        L_vcg=self.Ls_vcg_min[i], L_vcg_min=self.Ls_vcg_min[i], \
                        L_vcg_max=self.Ls_vcg_max[i], Lmax_stock=1, comb_vcg=32, \
                        gamma_2m=self.gamma_2m, gamma_d=self.gamma_2m/2, \
                        include_tetraplexes=self.include_tetraplexes)
                cmplxs.identify_reaction_types_cvfolp_all_complexes()

                for rt in cmplxs.index_2_react_type_cvflvs_simpnot.values():
                    if not rt in self.reaction_type_cvflvs_2_index:
                        if len(self.reaction_type_cvflvs_2_index)!=0:
                            last_key = next(reversed(self.reaction_type_cvflvs_2_index))
                            last_index = self.reaction_type_cvflvs_2_index[last_key]
                            self.reaction_type_cvflvs_2_index[rt] = last_index + 1
                        else:
                            self.reaction_type_cvflvs_2_index[rt] = 0

                for rt in cmplxs.react_type_cvfolp_2_index.keys():
                    if not rt in self.reaction_type_cvfolp_2_index:
                        if len(self.reaction_type_cvfolp_2_index)!=0:
                            last_key = next(reversed(self.reaction_type_cvfolp_2_index))
                            last_index = self.reaction_type_cvfolp_2_index[last_key]
                            self.reaction_type_cvfolp_2_index[rt] = last_index + 1
                        else:
                            self.reaction_type_cvfolp_2_index[rt] = 0
        
        elif self.variable == 'kappa':
            cmplxs = ComplexConstructor(l_unique=3, alphabet=4, \
                    L_vcg=self.Ls_vcg_min[0], L_vcg_min=self.Ls_vcg_min[0], \
                    L_vcg_max=self.Ls_vcg_max[0], Lmax_stock=1, comb_vcg=32, \
                    gamma_2m=self.gamma_2m, gamma_d=self.gamma_2m/2, \
                    include_tetraplexes=self.include_tetraplexes)
            cmplxs.identify_reaction_types_cvfolp_all_complexes()

            for rt in cmplxs.index_2_react_type_cvflvs_simpnot.values():
                if not rt in self.reaction_type_cvflvs_2_index:
                    if len(self.reaction_type_cvflvs_2_index)!=0:
                        last_key = next(reversed(self.reaction_type_cvflvs_2_index))
                        last_index = self.reaction_type_cvflvs_2_index[last_key]
                        self.reaction_type_cvflvs_2_index[rt] = last_index + 1
                    else:
                        self.reaction_type_cvflvs_2_index[rt] = 0

            for rt in cmplxs.react_type_cvfolp_2_index.keys():
                if not rt in self.reaction_type_cvfolp_2_index:
                    if len(self.reaction_type_cvfolp_2_index)!=0:
                        last_key = next(reversed(self.reaction_type_cvfolp_2_index))
                        last_index = self.reaction_type_cvfolp_2_index[last_key]
                        self.reaction_type_cvfolp_2_index[rt] = last_index + 1
                    else:
                        self.reaction_type_cvfolp_2_index[rt] = 0


    def construct_single_data_point(self, index):

        dp = DataPoint(L_vcg_min=self.Ls_vcg_min[index], L_vcg_max=self.Ls_vcg_max[index], \
                       Lambda_vcg=self.Lambdas_vcg[index], gamma_2m=self.gamma_2m, \
                       c0_stock_all=self.c0_stock_all)
        
        return dp


    def construct_all_data_points(self):

        self.dps: List[DataPoint] = []
        
        p = Pool(os.cpu_count()-4)
        indices = np.arange(len(self.Ls_vcg_max))
        for out in tqdm.tqdm(p.imap(self.construct_single_data_point, indices), \
                                total=len(indices)):
            self.dps.append(out)
        p.close()


    def extract_ratios_reaction_types_cvflvs_from_data_points(self):

        self.ratios_nuc_cvflvs = {key:[] for key in self.reaction_type_cvflvs_2_index}

        for dp in self.dps:
            for rt in self.ratios_nuc_cvflvs.keys():
                if rt in dp.ratios_nuc_cvflvs:
                    self.ratios_nuc_cvflvs[rt].append(dp.ratios_nuc_cvflvs[rt])
                else:
                    self.ratios_nuc_cvflvs[rt].append(0.)
        
        self.ratios_nuc_cvflvs = {key:np.asarray(values) for key, values in \
                                  self.ratios_nuc_cvflvs.items()}
    

    def extract_ratios_reaction_types_cvfolp_from_data_points(self):

        self.ratios_nuc_cvfolp = {key:[] for key in self.reaction_type_cvfolp_2_index}

        for dp in self.dps:
            for rt in self.ratios_nuc_cvfolp.keys():
                if rt in dp.ratios_nuc_cvfolp:
                    self.ratios_nuc_cvfolp[rt].append(dp.ratios_nuc_cvfolp[rt])
                else:
                    self.ratios_nuc_cvfolp[rt].append(0.)

        self.ratios_nuc_cvfolp = {key:np.asarray(values) for key, values in \
                                  self.ratios_nuc_cvfolp.items()}

    
    def extract_errorfree_ratios_cvflvs(self):
        self.errorfree_ratios_cvflvs = np.asarray([dp.errorfree_ratio_cvflvs_opt \
                                                  for dp in self.dps])
        
    
    def plot_fraction_incorporated_nucleotides_by_type(self, f, ax, y_values, \
                                                       threshold_rest=3e-3):

        if (f is None) and (ax is None):
            f, ax = plt.subplots(1,1,figsize=(9,3.2))

        react_type_cvflvs_2_colors = {'ff_v_f_c':sns.color_palette("colorblind")[0], \
                                      'fv_v_v_c':sns.color_palette("colorblind")[1], \
                                      'fv_v_v_f':sns.color_palette("colorblind")[2], \
                                      'vv_v_v_c':sns.color_palette("colorblind")[3], \
                                      'vv_v_v_f':sns.color_palette("colorblind")[4]
                                      }
        alpha_transparency = 0.6
        # switch tex-mode off
        plt.rcParams.update(
            {'text.usetex':False}
        )


        if y_values=='lvcgmax':
            for dp in self.dps:
                dp.identify_relevant_subset_of_ratios_nuc_cvfolp(threshold_rest)
                dp.map_cvfolp_to_cvflvs()
                ratios_nuc_cvfolp_cum = np.cumsum(np.asarray(list(dp.ratios_nuc_cvfolp_rel.values())))
                counter = 0
                for i, (key, value) in enumerate(dp.ratios_nuc_cvfolp_rel.items()):
                    if value >= 0.04 and key != 'rest':
                        rt_cvflvs = dp.react_type_cvfolp_2_react_type_cvflvs[key]
                        color=react_type_cvflvs_2_colors[rt_cvflvs]
                        p = ax.barh(dp.L_vcg_max, width=value, left=ratios_nuc_cvfolp_cum[i]-value, \
                            color=color, edgecolor='black', linewidth=0.75, alpha=alpha_transparency)
                        lt, le1, le2 = eval(key.split('_')[0])
                        cvf = key.split('_')[1]
                        plt.rcParams.update(
                            {'text.usetex':True}
                        )
                        label = r'$\frac{%s|%s}{%s}$' %(le1, le2, lt)
                        ax.bar_label(p, labels=[label], label_type='center', color='black')
                        plt.rcParams.update(
                            {'text.usetex':False}
                        )
                    elif value < 0.04 and key != 'rest':
                        rt_cvflvs = dp.react_type_cvfolp_2_react_type_cvflvs[key]
                        color=react_type_cvflvs_2_colors[rt_cvflvs]
                        ax.barh(dp.L_vcg_max, width=value, left=ratios_nuc_cvfolp_cum[i]-value, \
                                color=color, edgecolor='black', linewidth=0.75, alpha=alpha_transparency)
                    elif key == 'rest':
                        ax.barh(dp.L_vcg_max, width=value, left=ratios_nuc_cvfolp_cum[i]-value, \
                                color='white', edgecolor='black', linewidth=0.75, alpha=alpha_transparency, \
                                hatch='///')
            ax.set_xlim([-0.015, 1.015])
            # ax.set_xlabel('fraction of incorporated nucleotides')
            ax.set_xlabel('ligation share $s$')
            ax.set_ylabel('maximal length of\n' \
                          + 'VCG oligomers $L_\mathrm{V}^\mathrm{max}$ (nt)')

            # add legend
            react_type_2_human_readable = {
                'ff_v_f_c': '(F+F)', \
                'fv_v_v_c': '(F+V), c', \
                'fv_v_v_f': '(F+V), f', \
                'vv_v_v_c': '(V+V), c', \
                'vv_v_v_f': '(V+V), f'
            }

            handles = []
            for rt, color in react_type_cvflvs_2_colors.items():
                pop = mpl.patches.Patch(color=color, label=react_type_2_human_readable[rt], alpha=alpha_transparency)
                handles.append(pop)
            ax.legend(handles=handles, bbox_to_anchor=(0.5, 1.03), loc='lower center', ncols=5, \
                bbox_transform=ax.transAxes)
    
        elif y_values=='lvcgmin':
            for dp in self.dps:
                dp.identify_relevant_subset_of_ratios_nuc_cvfolp(threshold_rest)
                dp.map_cvfolp_to_cvflvs()
                ratios_nuc_cvfolp_cum = np.cumsum(np.asarray(list(dp.ratios_nuc_cvfolp_rel.values())))
                counter = 0
                for i, (key, value) in enumerate(dp.ratios_nuc_cvfolp_rel.items()):
                    if value >= 0.04 and key != 'rest':
                        rt_cvflvs = dp.react_type_cvfolp_2_react_type_cvflvs[key]
                        color=react_type_cvflvs_2_colors[rt_cvflvs]
                        p = ax.barh(dp.L_vcg_min, width=value, left=ratios_nuc_cvfolp_cum[i]-value, \
                            color=color, edgecolor='black', linewidth=0.75, alpha=alpha_transparency)
                        lt, le1, le2 = eval(key.split('_')[0])
                        cvf = key.split('_')[1]
                        plt.rcParams.update(
                            {'text.usetex':True}
                        )
                        label = r'$\frac{%s|%s}{%s}$' %(le1, le2, lt)
                        ax.bar_label(p, labels=[label], label_type='center', color='black')
                        plt.rcParams.update(
                            {'text.usetex':False}
                        )
                    elif value < 0.04 and key != 'rest':
                        rt_cvflvs = dp.react_type_cvfolp_2_react_type_cvflvs[key]
                        color=react_type_cvflvs_2_colors[rt_cvflvs]
                        ax.barh(dp.L_vcg_min, width=value, left=ratios_nuc_cvfolp_cum[i]-value, \
                                color=color, edgecolor='black', linewidth=0.75, alpha=alpha_transparency)
                    elif key == 'rest':
                        ax.barh(dp.L_vcg_min, width=value, left=ratios_nuc_cvfolp_cum[i]-value, \
                                color='white', edgecolor='black', linewidth=0.75, alpha=alpha_transparency, \
                                hatch='///')
            ax.set_xlim([-0.015, 1.015])
            ax.set_xlabel('ligation share $s$')
            ax.set_ylabel('minimal length of\n' \
                          + 'VCG oligomers $L_\mathrm{V}^\mathrm{min}$ (nt)')

            # add legend
            react_type_2_human_readable = {
                'ff_v_f_c': '(F+F)', \
                'fv_v_v_c': '(F+V), c', \
                'fv_v_v_f': '(F+V), f', \
                'vv_v_v_c': '(V+V), c', \
                'vv_v_v_f': '(V+V), f'
            }
            
            handles = []
            for rt, color in react_type_cvflvs_2_colors.items():
                pop = mpl.patches.Patch(color=color, label=react_type_2_human_readable[rt], alpha=alpha_transparency)
                handles.append(pop)
            ax.legend(handles=handles, bbox_to_anchor=(0.5, 1.03), loc='lower center', ncols=5, \
                bbox_transform=ax.transAxes)

# flags to control what is done
READ = False # decide whether to read from pre-computed output files
# pre-computed output files are not part of the repo, so run the calculation
WRITE_Lvcgmax = False
WRITE_kappavar = False
WRITE_Lvcgmin3_Lvcgmaxvar = False
WRITE_Lvcgminvar_Lvcgmax10 = False
WRITE = any([WRITE_Lvcgmax,WRITE_kappavar,\
             WRITE_Lvcgminvar_Lvcgmax10,WRITE_Lvcgmin3_Lvcgmaxvar])
PLOT = True # uses the data to produce plots

if WRITE:
    # compute the data
    if WRITE_Lvcgmax:
        Ls_vcg_max = np.arange(3, 11, 1)
        print("calculate ds_Lvcgminmaxvar")
        ds_Lvcgminmaxvar = DataSet(Ls_vcg_min=Ls_vcg_max, Ls_vcg_max=Ls_vcg_max, \
                                  Lambdas_vcg=np.inf*np.ones(len(Ls_vcg_max)), \
                                   variable='lvcgminmax', \
                                   label=r'$L_\mathrm{VCG}^\mathrm{min} = L_\mathrm{VCG}^\mathrm{max}$', \
                                   gamma_2m=-2.5, c0_stock_all=1e-4)
        print("save ds_Lvcgminmaxvar to pickle")
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgminmaxvar.pkl', 'wb')
        pkl.dump(ds_Lvcgminmaxvar, f)
        f.close()

        Ls_vcg_max = np.arange(5, 11, 1)
        print("calculate ds_Lvcgmaxvar_Lvcgmin3")
        ds_Lvcgmaxvar_Lvcgmin3 = DataSet(Ls_vcg_min=3*np.ones(len(Ls_vcg_max), dtype=int), \
                                         Ls_vcg_max=Ls_vcg_max, \
                                         Lambdas_vcg=np.inf*np.ones(len(Ls_vcg_max)), \
                                         variable='lvcgmax', \
                                         label=r'$L_\mathrm{VCG}^\mathrm{min}$ = 3', \
                                         gamma_2m=-2.5, c0_stock_all=1e-4)
        print("save ds_Lvcgmaxvar_Lvcgmin3 to pickle")
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgmaxvar_Lvcgmin3.pkl', 'wb')
        pkl.dump(ds_Lvcgmaxvar_Lvcgmin3, f)
        f.close()

        print("calculate ds_Lvcgmaxvar_Lvcgmin5")
        ds_Lvcgmaxvar_Lvcgmin5 = DataSet(Ls_vcg_min=5*np.ones(len(Ls_vcg_max), dtype=int), \
                                         Ls_vcg_max=Ls_vcg_max, \
                                         variable='lvcgmax', \
                                         label=r'$L_\mathrm{VCG}^\mathrm{min}$ = 5', \
                                         Lambdas_vcg=np.inf*np.ones(len(Ls_vcg_max)), \
                                         gamma_2m=-2.5, c0_stock_all=1e-4)
        print("save ds_Lvcgmaxvar_Lvcgmin5 to pickle")
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgmaxvar_Lvcgmin5.pkl', 'wb')
        pkl.dump(ds_Lvcgmaxvar_Lvcgmin5, f)
        f.close()

        print("calculate ds_Lvcgmaxvar_Lvcgmin7")
        ds_Lvcgmaxvar_Lvcgmin7 = DataSet(Ls_vcg_min=7*np.ones(len(Ls_vcg_max[2:]), dtype=int), \
                                         Ls_vcg_max=Ls_vcg_max[2:], \
                                         Lambdas_vcg=np.inf*np.ones(len(Ls_vcg_max)-2), \
                                         variable='lvcgmax', \
                                         label=r'$L_\mathrm{VCG}^\mathrm{min}$ = 7', \
                                         gamma_2m=-2.5, c0_stock_all=1e-4)
        print("save ds_Lvcgmaxvar_Lvcgmin7 to pickle")
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgmaxvar_Lvcgmin7.pkl', 'wb')
        pkl.dump(ds_Lvcgmaxvar_Lvcgmin5, f)
        f.close()
        
        dss_Lvcgmax = [ds_Lvcgminmaxvar, ds_Lvcgmaxvar_Lvcgmin3, \
                       ds_Lvcgmaxvar_Lvcgmin5, ds_Lvcgmaxvar_Lvcgmin7]

    if WRITE_kappavar:
        kappas_vcg = np.linspace(-4.5, 4.5, 25)
        print("calculate ds_kappavar_Lvcgmin3")
        ds_kappavar_Lvcgmin3 = DataSet(Ls_vcg_min=3*np.ones(len(kappas_vcg), dtype=int), \
                                       Ls_vcg_max=10*np.ones(len(kappas_vcg), dtype=int), \
                                       Lambdas_vcg=1/kappas_vcg, variable='kappa', \
                                       label=r'$L_\mathrm{VCG}^\mathrm{min}$ = 3', \
                                       gamma_2m=-2.5, c0_stock_all=1e-4)
        print("save ds_kappavar_Lvcgmin3 to pickle")
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_kappavar_Lvcgmin3.pkl', 'wb')
        pkl.dump(ds_kappavar_Lvcgmin3, f)
        f.close()

        print("calculate ds_kappavar_Lvcgmin5")
        ds_kappavar_Lvcgmin5 = DataSet(Ls_vcg_min=5*np.ones(len(kappas_vcg), dtype=int), \
                                       Ls_vcg_max=10*np.ones(len(kappas_vcg), dtype=int), \
                                       Lambdas_vcg=1/kappas_vcg, variable='kappa', \
                                       label=r'$L_\mathrm{VCG}^\mathrm{min}$ = 5', \
                                       gamma_2m=-2.5, c0_stock_all=1e-4)
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_kappavar_Lvcgmin5.pkl', 'wb')
        pkl.dump(ds_kappavar_Lvcgmin5, f)
        f.close()
        
        print("calculate ds_kappavar_Lvcgmin7")
        ds_kappavar_Lvcgmin7 = DataSet(Ls_vcg_min=7*np.ones(len(kappas_vcg), dtype=int), \
                                       Ls_vcg_max=10*np.ones(len(kappas_vcg), dtype=int), \
                                       Lambdas_vcg=1/kappas_vcg, variable='kappa', \
                                       label=r'$L_\mathrm{VCG}^\mathrm{min}$ = 7', \
                                       gamma_2m=-2.5, c0_stock_all=1e-4)
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_kappavar_Lvcgmin7.pkl', 'wb')
        pkl.dump(ds_kappavar_Lvcgmin7, f)
        f.close()

        dss_kappasvcg = [ds_kappavar_Lvcgmin3, ds_kappavar_Lvcgmin5, ds_kappavar_Lvcgmin7]
    
    if WRITE_Lvcgminvar_Lvcgmax10:
        Ls_vcg_min = np.array([3,4,5,6,7,8])
        print("calculate ds_Lvcgminvar_Lvcgmax10")
        ds_Lvcgminvar_Lvcgmax10 = DataSet(Ls_vcg_min=Ls_vcg_min, \
                                          Ls_vcg_max=10*np.ones(len(Ls_vcg_min), dtype=int), \
                                          Lambdas_vcg=np.inf*np.ones(len(Ls_vcg_min)), \
                                          variable='lvcgmin', \
                                          label=None, gamma_2m=-2.5, c0_stock_all=1e-4)

        print("write the data to a pickle file")
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgminvar_Lvcgmax10.pkl', 'wb')
        pkl.dump(ds_Lvcgminvar_Lvcgmax10, f)
        f.close()

    if WRITE_Lvcgmin3_Lvcgmaxvar: 
        Ls_vcg_max = np.array([5,6,7,8,9,10])
        print("calculate ds_Lvcgmin3_Lvcgmaxvar")
        ds_Lvcgmin3_Lvcgmaxvar = DataSet(Ls_vcg_min=3*np.ones(len(Ls_vcg_max), dtype=int), \
                                         Ls_vcg_max=Ls_vcg_max, \
                                         Lambdas_vcg=np.inf*np.ones(len(Ls_vcg_max)), \
                                         variable='lvcgmax', \
                                         label=None, gamma_2m=-2.5, c0_stock_all=1e-4)

        print("write the data to a pickle file")
        f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgmin3_Lvcgmaxvar.pkl', 'wb')
        pkl.dump(ds_Lvcgmin3_Lvcgmaxvar, f)
        f.close()

if READ:
    print("load data")
    
    f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgminmaxvar.pkl', 'rb')
    ds_Lvcgminmaxvar = pkl.load(f)
    f.close()

    f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgmaxvar_Lvcgmin3.pkl', 'rb')
    ds_Lvcgmaxvar_Lvcgmin3 = pkl.load(f)
    f.close()

    f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgmaxvar_Lvcgmin5.pkl', 'rb')
    ds_Lvcgmaxvar_Lvcgmin5 = pkl.load(f)
    f.close()

    f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgmaxvar_Lvcgmin7.pkl', 'rb')
    ds_Lvcgmaxvar_Lvcgmin7 = pkl.load(f)
    f.close()

    dss_Lvcgmax = [ds_Lvcgminmaxvar, ds_Lvcgmaxvar_Lvcgmin3, \
                   ds_Lvcgmaxvar_Lvcgmin5, ds_Lvcgmaxvar_Lvcgmin7]
    
    f = open('../../outputs/changing_LVmin_LVmax/data/ds_kappavar_Lvcgmin3.pkl', 'rb')
    ds_kappavar_Lvcgmin3 = pkl.load(f)
    f.close()

    f = open('../../outputs/changing_LVmin_LVmax/data/ds_kappavar_Lvcgmin5.pkl', 'rb')
    ds_kappavar_Lvcgmin5 = pkl.load(f)
    f.close()

    f = open('../../outputs/changing_LVmin_LVmax/data/ds_kappavar_Lvcgmin7.pkl', 'rb')
    ds_kappavar_Lvcgmin7 = pkl.load(f)
    f.close()

    dss_kappasvcg = [ds_kappavar_Lvcgmin3, ds_kappavar_Lvcgmin5, ds_kappavar_Lvcgmin7]

    f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgminvar_Lvcgmax10.pkl', 'rb')
    ds_Lvcgminvar_Lvcgmax10 = pkl.load(f)
    f.close()

    f = open('../../outputs/changing_LVmin_LVmax/data/ds_Lvcgmin3_Lvcgmaxvar.pkl', 'rb')
    ds_Lvcgmin3_Lvcgmaxvar = pkl.load(f)
    f.close()

if PLOT:
    print("plot")
    f, axs = plt.subplot_mosaic([['A','B','C'], \
                                 ['D','D','D'], \
                                 ['E','E','E']], figsize=(3*4.5,3*3.4))
    
    # add names to the panels
    for letter in ['A','B','C']:
        axs[letter].text(-0.13, 1.15, letter, transform=axs[letter].transAxes,
          fontsize=18, fontweight='bold', va='top', ha='right')
    axs['D'].text(-0.13/3, 1.2, 'D', transform=axs['D'].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')
    axs['E'].text(-0.13/3, 1.2, 'E', transform=axs['E'].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')

    axs['A'].bar([1,], [1e-4], color='grey')
    axs['A'].bar(np.array([4,5,6,7,8,9])-0.2, 0.5*np.array([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]), color='grey', width=0.4)
    axs['A'].bar(np.array([4,5,6,7,8,9])+0.2, 3e-6*np.exp(-0.5*np.array([0,1,2,3,4,5])), \
                                                          color='grey', width=0.4, alpha=0.5)
    axs['A'].plot(np.array([4,5,6,7,8,9]), 4e-6*np.exp(-0.5*np.array([0,1,2,3,4,5])), \
                                                          color='grey', alpha=0.5, linestyle='dashed')
    axs['A'].vlines(x=4, ymin=0, ymax=1, linestyle='dashed', color='grey')
    t1 = axs['A'].text(x=4.1, y=3e-5, s=r'$L_\mathrm{V}^\mathrm{min}$', \
                  rotation=90, horizontalalignment='center')
    t1.set_bbox(dict(facecolor='white', edgecolor='white'))
    axs['A'].vlines(x=9, ymin=0, ymax=1, linestyle='dashed', color='grey')
    t2 = axs['A'].text(x=9.1, y=3e-5, s=r'$L_\mathrm{V}^\mathrm{max}$', \
                  rotation=90, horizontalalignment='center')
    t2.set_bbox(dict(facecolor='white', edgecolor='white'))
    axs['A'].annotate('', xy=(5,1e-5),xytext=(4,1e-5), \
                      arrowprops=dict(facecolor='grey', shrink=0., edgecolor='grey'))
    axs['A'].annotate('', xy=(3,1e-5),xytext=(4,1e-5), \
                      arrowprops=dict(facecolor='grey', shrink=0., edgecolor='grey'))
    axs['A'].annotate('', xy=(10,1e-5),xytext=(9,1e-5), \
                      arrowprops=dict(facecolor='grey', shrink=0., edgecolor='grey'))
    axs['A'].annotate('', xy=(8,1e-5),xytext=(9,1e-5), \
                      arrowprops=dict(facecolor='grey', shrink=0., edgecolor='grey'))
    axs['A'].text(x=6, y=2e-6, s=r'$\sim \kappa_\mathrm{V}^{-1}$', color='grey', alpha=1.0)
    axs['A'].set_yscale('log')
    axs['A'].set_ylim([1e-7, 1.3e-4])
    axs['A'].set_xlim([0.2, 10.8])
    axs['A'].set_xlabel(r'oligomer length $L$ (nt)')
    axs['A'].set_ylabel(r'concentration $c(L)$ (M)')

    ds_Lvcgminmaxvar = dss_Lvcgmax[0]
    map_labels = {r'$L_\mathrm{VCG}^\mathrm{min} = L_\mathrm{VCG}^\mathrm{max}$':\
                    r'$L_\mathrm{V}^\mathrm{min} = L_\mathrm{V}^\mathrm{max}$', \
                  r'$L_\mathrm{VCG}^\mathrm{min}$ = 3':\
                    r'$L_\mathrm{V}^\mathrm{min}$ = 3 nt', \
                  r'$L_\mathrm{VCG}^\mathrm{min}$ = 5':\
                    r'$L_\mathrm{V}^\mathrm{min}$ = 5 nt', \
                  r'$L_\mathrm{VCG}^\mathrm{min}$ = 7':\
                    r'$L_\mathrm{V}^\mathrm{min}$ = 7 nt'}
    axs['B'].plot(ds_Lvcgminmaxvar.Ls_vcg_max[2:], ds_Lvcgminmaxvar.errorfree_ratios_cvflvs[2:], \
                  label=map_labels[ds_Lvcgminmaxvar.label], color="C0")
    
    for i, ds in enumerate(dss_Lvcgmax[1:]):
        axs['B'].plot(ds.Ls_vcg_max, ds.errorfree_ratios_cvflvs, label=map_labels[ds.label], color=f"C{i+1}")
        L_vcg_min = ds.Ls_vcg_min[0]
                                  
    axs['B'].set_xlabel(r'maximal length of VCG oligomers $L_\mathrm{V}^\mathrm{max}$ (nt)')
    axs['B'].set_ylabel(r'replication efficiency $\eta_\mathrm{max}$')
    axs['B'].legend()
    
    for i, ds in enumerate(dss_kappasvcg):
        axs['C'].plot(1/ds.Lambdas_vcg, ds.errorfree_ratios_cvflvs, label=map_labels[ds.label])
        L_vcg_min = ds.Ls_vcg_min[0]
        index = np.where(ds_Lvcgminmaxvar.Ls_vcg_min == L_vcg_min)
        axs['C'].hlines(ds_Lvcgminmaxvar.errorfree_ratios_cvflvs[index], \
                       xmin=1.75, xmax=4.5, linestyle='dashed', color=f'C{i}')
    axs['C'].set_xlabel(r'inverse length scale $\kappa_\mathrm{V}$ (nt$^{-1}$)')
    axs['C'].set_ylabel(r'replication efficiency $\eta_\mathrm{max}$')
    
    t1 = axs['C'].text(-4.4, 0.25, r'sharp peak at $L^\mathrm{max}_\mathrm{V}$', rotation=90)
    t1.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    axs['C'].axvline(0, ymin=0, ymax=1, color='grey', linestyle='dashed')
    t2 = axs['C'].text(-0.25, 0.65, r'uniform', rotation=90)
    t2.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    t3 = axs['C'].text(4.0, 0.25, r'sharp peak at $L^\mathrm{min}_\mathrm{V}$', rotation=90)
    t3.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    axs['C'].legend(bbox_to_anchor=((.42, .12, .2, .4)))
    
    f.tight_layout()
    
    ds_Lvcgminvar_Lvcgmax10.plot_fraction_incorporated_nucleotides_by_type(f, axs['D'], \
                                                                           y_values='lvcgmin')
    
    ds_Lvcgmin3_Lvcgmaxvar.plot_fraction_incorporated_nucleotides_by_type(f, axs['E'], \
                                                                          y_values='lvcgmax')
    
    f.savefig(f'../../outputs/changing_LVmin_LVmax/main__influence_LVmin_LVmax__cfeedall_{1e-4:1.2e}.pdf', \
              bbox_inches='tight')
