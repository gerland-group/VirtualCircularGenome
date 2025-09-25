#!/bin/env python3

import sys
sys.path.append('../../src/')
import pickle as pkl
import tqdm
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['font.size'] = 12.0

from ComplexConstructor import *
from ConcentrationComputer import *

class SynergyAnalyzer:

    def __init__(self, Ls, gamma, include_tetraplexes=True):

        self.Ls = Ls
        self.cmplxs = ComplexConstructor(\
            l_unique=3, alphabet=4, \
            L_vcg=3, L_vcg_min=None, L_vcg_max=None, Lmax_stock=1, Ls=Ls, \
            comb_vcg=32, gamma_2m=gamma, gamma_d=gamma/2, \
            include_tetraplexes=include_tetraplexes)
        self.cmplxs.identify_reaction_types_cvfol_all_complexes()
        self.cmplxs.identify_reaction_types_cvfolp_all_complexes()

        self.c_1 = 1e-4
        N_points = 30
        self.cs_a = np.logspace(-8, -4, N_points-1)
        self.cs_b = np.logspace(-8, -4, N_points)
        self.create_sets_of_concentrations()

    def create_sets_of_concentrations(self):
        self.cs_sets = []
        self.indices_sets = []
        for i in range(len(self.cs_a)):
            for j in range(len(self.cs_b)):
                self.cs_sets.append((self.cs_a[i], self.cs_b[j]))
                self.indices_sets.append((i,j))
    
    def compute_fidelity__single_concentration_set(self, indices):

        index_a, index_b = indices
        cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                cs_ss_comb_tot=np.array([self.c_1, self.cs_a[index_a], self.cs_b[index_b]]))
        cncs.compute_equilibrium_concentration_log()
        cncs.compute_errorfree_ratio_cvflvs_exact()
        return index_a, index_b, cncs.errorfree_ratio_cvflvs_exact
        
    def compute_fidelity__all_concentration_sets(self):

        self.fidelities = np.nan*np.ones((len(self.cs_a), len(self.cs_b)))
        p = Pool(20)
        for out in tqdm.tqdm(p.imap(self.compute_fidelity__single_concentration_set, \
                                    self.indices_sets), \
                             total=len(self.indices_sets)):
            self.fidelities[out[0],out[1]] = out[2]
    
    def compute_fidelity__single_concentration_set__cslog(self, cslog, sign=1.0):

        cs = np.exp(cslog)
        print(cs)
        cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                cs_ss_comb_tot=np.array([self.c_1, cs[0], cs[1]]))
        cncs.compute_equilibrium_concentration_log()
        cncs.compute_errorfree_ratio_cvflvs_exact()
        return sign*cncs.errorfree_ratio_cvflvs_exact

    def maximize_fidelity(self):

        out = minimize(self.compute_fidelity__single_concentration_set__cslog, \
                       x0=np.log(np.array([1e-7,1e-7])), \
                       args=(-1.0,))
        return out
        

    def compute_cs_nuc_cvflvs__single_concentration_set(self, indices):

        index_a, index_b = indices
        cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                cs_ss_comb_tot=np.array([self.c_1, self.cs_a[index_a], self.cs_b[index_b]]))
        cncs.compute_equilibrium_concentration_log()
        cncs.compute_concentrations_added_nucleotides_productive_cvflvs()
        return index_a, index_b, cncs.cs_nuc_cvflvs
    
    def compute_cs_nuc_cvflvs__all_concentration_sets(self):

        self.css_nuc_cvflvs = np.nan*np.ones( (len(self.cmplxs.react_type_cvflvs_2_index), len(self.cs_a), len(self.cs_b)) )
        p = Pool(20)
        for out in tqdm.tqdm(p.imap(self.compute_cs_nuc_cvflvs__single_concentration_set, \
                                    self.indices_sets), \
                             total=len(self.indices_sets)):
            self.css_nuc_cvflvs[:,out[0],out[1]] = np.asarray(list(out[2].values()))
        p.close()

    def compute_cs_nuc_cvfol__single_concentration_sets(self, indices):

        index_a, index_b = indices
        cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                cs_ss_comb_tot=np.array([self.c_1, self.cs_a[index_a], self.cs_b[index_b]]))
        cncs.compute_equilibrium_concentration_log()
        cncs.compute_concentrations_added_nucleotides_productive_cvfol()
        return index_a, index_b, cncs.cs_nuc_cvfol
    
    def compute_cs_nuc_cvfol__all_concentration_sets(self):

        self.css_nuc_cvfol = np.nan*np.ones( (len(self.cmplxs.react_type_cvfol_2_index), len(self.cs_a), len(self.cs_b)) )
        p = Pool(20)
        for out in tqdm.tqdm(p.imap(self.compute_cs_nuc_cvfol__single_concentration_sets, \
                                    self.indices_sets), \
                             total=len(self.indices_sets)):
            self.css_nuc_cvfol[:,out[0],out[1]] = np.asarray(list(out[2].values()))
        p.close()

    def compute_cs_nuc_cvfolp__single_concentration_sets(self, indices):

        index_a, index_b = indices
        cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                cs_ss_comb_tot=np.array([self.c_1, self.cs_a[index_a], self.cs_b[index_b]]))
        cncs.compute_equilibrium_concentration_log()
        cncs.compute_concentrations_added_nucleotides_productive_cvfolp()
        return index_a, index_b, cncs.cs_nuc_cvfolp
    
    def compute_cs_nuc_cvfolp__all_concentration_sets(self):

        self.css_nuc_cvfolp = np.nan*np.ones( (len(self.cmplxs.react_type_cvfolp_2_index), len(self.cs_a), len(self.cs_b)) )
        p = Pool(20)
        for out in tqdm.tqdm(p.imap(self.compute_cs_nuc_cvfolp__single_concentration_sets, \
                                    self.indices_sets), \
                             total=len(self.indices_sets)):
            self.css_nuc_cvfolp[:,out[0],out[1]] = np.asarray(list(out[2].values()))
        p.close()

    
            
    def compute_concentration_added_nucleotides_by_length_added_oligomer__single_concentration_set(\
            self, indices):
        
        # c_a, c_b = cs
        index_a, index_b = indices

        cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
            cs_ss_comb_tot=np.array([self.c_1, self.cs_a[index_a], self.cs_b[index_b]]))
        cncs.compute_equilibrium_concentration_log()
        cncs.compute_concentrations_added_nucleotides_productive_cvfol()

        cs_nuc_nl = {}
        for rt in cncs.cs_nuc_cvfol.keys():
            l_added_nucs = min(eval(rt.split('_')[0])[1:])
            if not l_added_nucs in cs_nuc_nl:
                cs_nuc_nl[l_added_nucs] = cncs.cs_nuc_cvfol[rt]
            else:
                cs_nuc_nl[l_added_nucs] += cncs.cs_nuc_cvfol[rt]
        
        # return cs_nuc_nl
        return index_a, index_b, cs_nuc_nl

    def compute_concentration_added_nucleotides_by_length_added_oligomer__all_concentration_sets(self):

        self.css_nuc_nl = np.ones((3, len(self.cs_a), len(self.cs_b)))

        p = Pool(20)
        for out in tqdm.tqdm(p.imap(self.compute_concentration_added_nucleotides_by_length_added_oligomer__single_concentration_set, \
                                    self.indices_sets), total=len(self.indices_sets)):
            self.css_nuc_nl[:,out[0],out[1]] = np.asarray(list(out[2].values()))
        p.close()


    def compute_difference_incorporated_monomers_tetramers__single_concentration_set(\
            self, cs):
        cs_nuc_nl = \
            self.compute_concentration_added_nucleotides_by_length_added_oligomer__single_concentration_set(cs)
        return cs_nuc_nl[1] - cs_nuc_nl[4]

    def compute_concentration_equal_incorporation_monomers_tetramers__single_concentration_set(self, c_a):

        compute_diff = \
            lambda c_b: \
            self.compute_difference_incorporated_monomers_tetramers__single_concentration_set(\
                [c_a, c_b])

        out = root_scalar(compute_diff, bracket=[self.cs_b[0], self.cs_b[-1]])
        if out.converged==True:
            return out.root
        else:
            raise ValueError('did not converge')
    
    
    def compute_concentration_equal_incorporation_monomers_tetramers__all_concentration_sets(self):

        self.cs_a_ei = np.logspace(-5.5, -4.9, 10)
        self.cs_b_ei = []

        for c_a in self.cs_a_ei:
            print(f"c_a: {c_a:1.3}")
            c_b = self.compute_concentration_equal_incorporation_monomers_tetramers__single_concentration_set(c_a)
            print(f"c_b: {c_b:1.3}")
            self.cs_b_ei.append(c_b)

    def plot_fidelity(self, f=None, ax=None):

        if (f is None) and (ax is None):
            f, ax = plt.subplots(1,1,figsize=(4.5,3.2))
        CS_a, CS_b = np.meshgrid(self.cs_a, self.cs_b)
        p1 = ax.pcolormesh(CS_a, CS_b, self.fidelities.T, shading='gouraud')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$c(%d)$ (M)' %self.Ls[1])
        ax.set_ylabel(r'$c(%d)$ (M)' %self.Ls[2])
        ax.set_xlim([self.cs_a[0], self.cs_a[-1]])
        ax.set_ylim([self.cs_b[0], self.cs_b[-1]])
        cbar = f.colorbar(p1, ax=ax, pad=0.)
        cbar.set_label('replication efficiency $\eta$')
    
    def plot_cs_nuc_cvflvs(self):

        N_panels = len(self.cmplxs.react_type_cvflvs_2_index)
        N_y = int(np.ceil(np.sqrt(N_panels)))
        N_x = int(np.ceil(N_panels/N_y))
        CS_a, CS_b = np.meshgrid(self.cs_a, self.cs_b)
        css_nuc_cvflvs_tot = np.sum(self.css_nuc_cvflvs, axis=0)

        f, axs = plt.subplots(N_y,N_x,figsize=(N_x*4.5, N_y*3.2))
        for rt, rt_index in self.cmplxs.react_type_cvflvs_2_index.items():
            axs[rt_index//N_x,rt_index%N_x].set_title(self.cmplxs.index_2_react_type_cvflvs_simpnot[rt_index])
            p = axs[rt_index//N_x,rt_index%N_x].pcolormesh(CS_a, CS_b, self.css_nuc_cvflvs[rt_index].T/css_nuc_cvflvs_tot.T)
            divider = make_axes_locatable(axs[rt_index//N_x,rt_index%N_x])
            cax = divider.append_axes('right', size='3%', pad=0.1)
            cbar = f.colorbar(p, cax=cax)
            axs[rt_index//N_x,rt_index%N_x].set_xscale('log')
            axs[rt_index//N_x,rt_index%N_x].set_yscale('log')
            axs[rt_index//N_x,rt_index%N_x].set_xlabel(r'$c(%d)$ (M)' %self.Ls[1])
            axs[rt_index//N_x,rt_index%N_x].set_ylabel(r'$c(%d)$ (M)' %self.Ls[2])
            axs[rt_index//N_x,rt_index%N_x].set_xlim([self.cs_a[0], self.cs_a[-1]])
            axs[rt_index//N_x,rt_index%N_x].set_ylim([self.cs_b[0], self.cs_b[-1]])
        
        f.tight_layout()
    
    
    def plot_cs_nuc_cvfol(self, save_fig=False):

        N_panels = len(self.cmplxs.react_type_cvfol_2_index)
        N_y = int(np.ceil(np.sqrt(N_panels)))
        N_x = int(np.ceil(N_panels/N_y))
        CS_a, CS_b = np.meshgrid(self.cs_a, self.cs_b)
        css_nuc_cvfol_tot = np.sum(self.css_nuc_cvfol, axis=0)

        f, axs = plt.subplots(N_y,N_x,figsize=(N_x*4.5, N_y*3.2))
        for rt, rt_index in self.cmplxs.react_type_cvfol_2_index.items():
            axs[rt_index//N_x,rt_index%N_x].set_title(rt)
            p = axs[rt_index//N_x,rt_index%N_x].pcolormesh(CS_a, CS_b, self.css_nuc_cvfol[rt_index].T/css_nuc_cvfol_tot.T)
            divider = make_axes_locatable(axs[rt_index//N_x,rt_index%N_x])
            cax = divider.append_axes('right', size='3%', pad=0.1)
            cbar = f.colorbar(p, cax=cax)
            axs[rt_index//N_x,rt_index%N_x].set_xscale('log')
            axs[rt_index//N_x,rt_index%N_x].set_yscale('log')
            axs[rt_index//N_x,rt_index%N_x].set_xlabel(r'$c(%d)$ (M)' %self.Ls[1])
            axs[rt_index//N_x,rt_index%N_x].set_ylabel(r'$c(%d)$ (M)' %self.Ls[2])
            axs[rt_index//N_x,rt_index%N_x].set_xlim([self.cs_a[0], self.cs_a[-1]])
            axs[rt_index//N_x,rt_index%N_x].set_ylim([self.cs_b[0], self.cs_b[-1]])
        
        f.tight_layout()

        if save_fig:
            f.savefig('./cs_nuc_cvfol.pdf')
    
    def plot_cs_nuc_cvfolp(self, save_fig=False):

        N_panels = len(self.cmplxs.react_type_cvfolp_2_index)
        N_y = int(np.ceil(np.sqrt(N_panels)))
        N_x = int(np.ceil(N_panels/N_y))
        CS_a, CS_b = np.meshgrid(self.cs_a, self.cs_b)
        css_nuc_cvfolp_tot = np.sum(self.css_nuc_cvfolp, axis=0)

        f, axs = plt.subplots(N_y,N_x,figsize=(N_x*4.5, N_y*3.2))
        for rt, rt_index in self.cmplxs.react_type_cvfolp_2_index.items():
            axs[rt_index//N_x,rt_index%N_x].set_title(rt)
            p = axs[rt_index//N_x,rt_index%N_x].pcolormesh(CS_a, CS_b, \
                            self.css_nuc_cvfolp[rt_index].T/css_nuc_cvfolp_tot.T)
            divider = make_axes_locatable(axs[rt_index//N_x,rt_index%N_x])
            cax = divider.append_axes('right', size='3%', pad=0.1)
            cbar = f.colorbar(p, cax=cax)
            axs[rt_index//N_x,rt_index%N_x].set_xscale('log')
            axs[rt_index//N_x,rt_index%N_x].set_yscale('log')
            axs[rt_index//N_x,rt_index%N_x].set_xlabel(r'$c(%d)$ (M)' %self.Ls[1])
            axs[rt_index//N_x,rt_index%N_x].set_ylabel(r'$c(%d)$ (M)' %self.Ls[2])
            axs[rt_index//N_x,rt_index%N_x].set_xlim([self.cs_a[0], self.cs_a[-1]])
            axs[rt_index//N_x,rt_index%N_x].set_ylim([self.cs_b[0], self.cs_b[-1]])
        
        f.tight_layout()

        if save_fig:
            f.savefig('./cs_nuc_cvfolp.pdf')

    
    def plot_contourplot_cs_nuc_cvfolp(self, f=None, ax=None):

        if (f is None) and (ax is None):
            f, ax = plt.subplots(1,1,figsize=(4.5,3.2))
        
        threshold = 0.2

        css_nuc_cvfolp_tot = np.sum(self.css_nuc_cvfolp, axis=0)
        CS_a, CS_b = np.meshgrid(self.cs_a, self.cs_b)
        number_contribs = 0
        for rt, rt_index in self.cmplxs.react_type_cvfolp_2_index.items():
            if np.any(self.css_nuc_cvfolp[rt_index]/css_nuc_cvfolp_tot >= threshold):
                number_contribs += 1
        
        counter = 0
        for rt, rt_index in self.cmplxs.react_type_cvfolp_2_index.items():
            if np.any(self.css_nuc_cvfolp[rt_index]/css_nuc_cvfolp_tot >= threshold):
                
                colors = mpl.colormaps['Set2'].colors
                #colors = sns.color_palette('colorblind')
                ax.contourf(CS_a, CS_b, self.css_nuc_cvfolp[rt_index].T/css_nuc_cvfolp_tot.T, \
                            levels=[threshold, 1], colors=[colors[counter]], \
                            alpha=0.6)
                L_t, L_e1, L_e2 = eval(rt.split('_')[0])
                cvf = 'c' if rt.split('_')[1] == 'c' else 'f'
                ax.bar([0.], [0.], label=r"$\frac{%d|%d}{%d}$, %s" %(L_e1, L_e2, L_t, cvf), \
                        color=colors[counter], alpha=0.6)
                counter += 1

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$c(%d)$ (M)' %self.Ls[1])
        ax.set_ylabel(r'$c(%d)$ (M)' %self.Ls[2])
        ax.set_xlim([self.cs_a[0], self.cs_a[-1]])
        ax.set_ylim([self.cs_b[0], self.cs_b[-1]])
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    def plot(self, save_fig, main_vs_SI):

        f, axs = plt.subplots(1,3,figsize=(3*4.6,3.4), constrained_layout=True)
        axs[0].text(-0.175, 1.15, 'A', transform=axs[0].transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')
        
        axs[0].bar([1], [1e-4], color='grey')
        for L in self.Ls:
            if not L == 1:
                axs[0].bar([L], [1e-6], color='grey')
                axs[0].annotate('', xy=(L,2e-6), xytext=(L,1e-6), \
                                arrowprops=dict(facecolor='grey', shrink=0., edgecolor='grey'))
                axs[0].annotate('', xy=(L,5e-7), xytext=(L,1e-6), \
                        arrowprops=dict(facecolor='grey', shrink=0., edgecolor='white'))
        axs[0].set_xticks([2,4,6,8,10])
        axs[0].set_yscale('log')
        axs[0].set_ylim([1e-7, 1.3e-4])
        axs[0].set_xlim([0.2, 10.8])
        axs[0].set_xlabel(r'oligomer length $L$ (nt)')
        axs[0].set_ylabel(r'concentration $c(L)$ (M)')

        axs[1].text(-0.175, 1.15, 'B', transform=axs[1].transAxes,
                    fontsize=18, fontweight='bold', va='top', ha='right')
        self.plot_fidelity(f, axs[1])
        axs[2].text(-0.175, 1.15, 'C', transform=axs[2].transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')
        self.plot_contourplot_cs_nuc_cvfolp(f, axs[2])
        # f.tight_layout()
        if save_fig:
            plt.savefig('../../outputs/changing_LVmin_LVmax/'\
                        +f'{main_vs_SI}__length_modulated_facilitated_ligation__Ls_{self.Ls}__'\
                        +f'cfeedall_{self.c_1:1.2e}__gamma{self.cmplxs.gamma_2m:1.2f}.pdf')
        plt.show()

################################# CREATE PLOTS #################################

sa1 = SynergyAnalyzer(Ls=[1,4,8], gamma=-2.5, include_tetraplexes=True)
sa1.compute_fidelity__all_concentration_sets()
sa1.compute_cs_nuc_cvfolp__all_concentration_sets()
sa1.plot(save_fig=True, main_vs_SI='main')

sa2 = SynergyAnalyzer(Ls=[1,4,8], gamma=-5, include_tetraplexes=True)
sa2.compute_fidelity__all_concentration_sets()
sa2.compute_cs_nuc_cvfolp__all_concentration_sets()
sa2.plot(save_fig=True, main_vs_SI='SI')

sa3 = SynergyAnalyzer(Ls=[1,7,8], gamma=-2.5, include_tetraplexes=True)
sa3.compute_fidelity__all_concentration_sets()
sa3.compute_cs_nuc_cvfolp__all_concentration_sets()
sa3.plot(save_fig=True, main_vs_SI='SI')

sa4 = SynergyAnalyzer(Ls=[1,7,8], gamma=-5., include_tetraplexes=True)
sa4.compute_fidelity__all_concentration_sets()
sa4.compute_cs_nuc_cvfolp__all_concentration_sets()
sa4.plot(save_fig=True, main_vs_SI='SI')
