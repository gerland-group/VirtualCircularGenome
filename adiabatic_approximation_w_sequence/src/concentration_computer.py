#!/bin/env python3

import matplotlib.pyplot as plt
import numba
import numpy as np
import pickle as pkl
import scipy.optimize
import time

from strand import *
from helix import *
from complex import *
from compound_container import *
from energy_calculator import *


@numba.njit
def compute_concentration_all_cmplxs_props(cs_strands, strand_indices_flat, \
        Kds_flat):

    cs_cmplxs = np.zeros(len(strand_indices_flat))

    for i in range(len(strand_indices_flat)):
        strand_indices_loc = strand_indices_flat[i]
        cs_cmplxs[i] = np.prod(cs_strands[strand_indices_loc])/Kds_flat[i]

    return cs_cmplxs


class ConcentrationComputer:

    def __init__(self, comp_cont):
        
        # compound container that includes all strands and complexes
        self.comp_cont = comp_cont

        # total concentration for each strand
        self.read_total_strand_concentration_from_compound_container()
    
    
    def read_total_strand_concentration_from_compound_container(self):

        self.cs_strands_tot = np.zeros(len(self.comp_cont.strands))

        for i, strand in enumerate(self.comp_cont.strands):
            self.cs_strands_tot[i] = strand.conc

        if(np.any(np.isnan(self.cs_strands_tot))):
            raise ValueError('invalid total concentrations')
            
    
    def compute_concentration_single_cmplx(self, cs_strands, cmplx):
        
        indices = [self.comp_cont.strandid2index[sid] for sid in \
                cmplx.strandids_simp]
        return np.prod(cs_strands[indices])/cmplx.Kd

    
    def compute_concentration_all_cmplxs(self, cs_strands):

        return compute_concentration_all_cmplxs_props(cs_strands, \
                self.comp_cont.strand_indices_flat, self.comp_cont.Kds_flat)


    def compute_mass_conservation(self, cs_strands):

        # for each strand, this function evaluates
        # diff = c_strand 
        #        + c_(cmplxs that include the strand)
        #        - c_(total concentration for strand)

        diff = np.copy(cs_strands)

        cs_cmplxs_flat = self.compute_concentration_all_cmplxs(cs_strands)
        
        for sindex in self.comp_cont.strandindex2cmplxindexflat.keys():
            cindices = self.comp_cont.strandindex2cmplxindexflat[sindex]
            c = np.sum(cs_cmplxs_flat[cindices])
            diff[sindex] += c

        diff -= self.cs_strands_tot
        diff /= self.cs_strands_tot
        # print(np.sum(diff**2))
        
        return diff

    def compute_equilibrium_concentration(self, atol=1e-15, rtol=0, cs_init=None):
        
        if cs_init is None:
            cs_init = self.cs_strands_tot
        
        # sol = scipy.optimize.root(self.compute_mass_conservation, \
        #         self.cs_strands_tot, method='lm', tol=1e-15)#, options={'disp':True})
        # sol = scipy.optimize.root(self.compute_mass_conservation, \
        #         self.cs_strands_tot, method='krylov', options={'fatol':tol, 'disp':True})

        # sol = scipy.optimize.root(self.compute_mass_conservation, \
        #         cs_init, method='df-sane', options={'ftol':rtol, 'fatol':atol, \
        #         'disp':True, 'sigma_eps':1e-3, 'maxfev':200})   
        sol = scipy.optimize.root(self.compute_mass_conservation, \
                cs_init, method='lm', options={'ftol':atol, 'verbose':1})
        if sol.success:
            self.success = True
            self.cs_strands_equ = sol.x
            self.cs_cmplxs_equ = self.compute_concentration_all_cmplxs(\
                    self.cs_strands_equ)

        else:
            self.success = False
            print('Equilibrium concentrations could not be computed via linear mass conservation. Root finding did not converge.')


    def compute_mass_conservation_log(self, cs_strands_log):

        cs_strands = np.exp(cs_strands_log)

        diff = np.copy(cs_strands)

        cs_cmplxs_flat = self.compute_concentration_all_cmplxs(cs_strands)
        
        for sindex in self.comp_cont.strandindex2cmplxindexflat.keys():
            cindices = self.comp_cont.strandindex2cmplxindexflat[sindex]
            c = np.sum(cs_cmplxs_flat[cindices])
            diff[sindex] += c

        diff -= self.cs_strands_tot
        diff /= self.cs_strands_tot
        print(np.sum(diff**2))
        
        return diff


    def compute_equilibrium_concentration_log(self, atol=1e-21, rtol=0, cs_init=None):

        if cs_init is None:
            cs_init = self.cs_strands_tot

        # sol = scipy.optimize.root(self.compute_mass_conservation, \
        #         self.cs_strands_tot, method='lm', tol=1e-15)#, options={'disp':True})
        # sol = scipy.optimize.root(self.compute_mass_conservation, \
        #         self.cs_strands_tot, method='krylov', options={'fatol':tol, 'disp':True})
        # sol = scipy.optimize.root(self.compute_mass_conservation_log, \
        #         np.log(cs_init), method='krylov', \
        #         options={'fatol':atol, 'disp':True, 'maxiter':300})
        sol = scipy.optimize.root(self.compute_mass_conservation_log, \
                np.log(cs_init), method='lm', options={'ftol':atol})
        
        if sol.success:
            self.success = True
            self.cs_strands_equ = np.exp(sol.x)
            self.cs_cmplxs_equ = self.compute_concentration_all_cmplxs(\
                    self.cs_strands_equ)
        else:
            self.success = False
            print('Equilibrium concentrations could not be computed via logarithmic mass conservation. Root finding did not converge.')


    def identify_concentrations_productive_complexes(self):

        cs_prod_inter = {key:0 for key in self.comp_cont.prods.keys()}

        for cmplx_class in self.comp_cont.prods.keys():
            for _, cmplxid in self.comp_cont.prods[cmplx_class]:
                cs_prod_inter[cmplx_class] += \
                    self.cs_cmplxs_equ[self.comp_cont.cmplxid2indexflat[cmplxid]]
                
        keys_sorted = sorted(cs_prod_inter.keys())
        self.cs_prod = {key:cs_prod_inter[key] for key in keys_sorted}


    def identify_concentrations_by_complex_type(self):

        cs_bytype_inter = {}

        for strandid, strandindex in self.comp_cont.strandid2index.items():
            hf.add_or_increase_in_dict((1,0), cs_bytype_inter, \
                    self.cs_strands_equ[strandindex])

        for cmplxid, cmplxindex in self.comp_cont.cmplxid2indexflat.items():

            nstrands = len(self.comp_cont.strand_indices_flat[cmplxindex])
            nmismatch = self.comp_cont.nmismatches_flat[cmplxindex]
            nmatch = self.comp_cont.nmatches_flat[cmplxindex]
        
            hf.add_or_increase_in_dict((nstrands, nmatch-nmismatch), cs_bytype_inter, \
                    self.cs_cmplxs_equ[cmplxindex])

        # sort the dictionary
        keys_sorted = sorted(list(cs_bytype_inter.keys()))
        self.cs_bytype = {key:cs_bytype_inter[key] for key in keys_sorted}


    def construct_sorted_indices_for_plotting(self):

        # construct index dictionary for the strands
        mass2strandindex = {}
        for i in self.comp_cont.strand_indices_sorted:
            hf.add_or_include_in_dict(self.comp_cont.strands[i].l, mass2strandindex, i)

        # construct dictionary that stores the properties of each cmplx
        # note: index convention is the "flat index convention" 
        # i. e. value in comp_cont.cmplxid2indexflat 
        mass2cmplxindexflat = {}
        mass2cmplxpropsflat = {}
        for i in self.comp_cont.cmplx_indices_flat_sorted:
            mass = self.comp_cont.props[i][0]
            hf.add_or_include_in_dict(mass, mass2cmplxindexflat, i)
            hf.add_or_include_in_dict(mass, mass2cmplxpropsflat, \
                    (self.comp_cont.props[i][1],self.comp_cont.props[i][2]))
        
        mass2cmplxprops_rel = {}
        for mass in mass2cmplxpropsflat.keys():
            props_loc = mass2cmplxpropsflat[mass]
            props_rel = {i:props_loc[i] for i in range(len(props_loc)) \
                         if props_loc[i-1] != props_loc[i]}
            if(len(props_rel)==0):
                props_rel = {0:props_loc[0]}
            mass2cmplxprops_rel[mass] = props_rel
        
        return mass2strandindex, mass2cmplxindexflat, mass2cmplxprops_rel



    def plot_equilibrium_concentrations(self, filepath=''):
        
        mass2strandindex, mass2cmplxindex, mass2cmplxprops_rel = \
                self.construct_sorted_indices_for_plotting()

        N = len(mass2strandindex)+len(mass2cmplxindex)
        Ny = int(np.ceil(np.sqrt(N)))
        Nx = int(np.ceil(N/Ny))

        f, ax = plt.subplots(Ny,Nx,figsize=(7.5*Nx,8*Ny), tight_layout=True)

        for im, mass in enumerate(list(mass2strandindex.keys())):

            # relevant concentration array
            cs = self.cs_strands_equ[mass2strandindex[mass]]
            
            ax[im//Nx,im%Nx].set_title('mass: %s' %mass, fontsize=15)
            ax[im//Nx,im%Nx].scatter(np.arange(len(cs)), cs)
            ax[im//Nx,im%Nx].set_xticks([0], ['0,0'], rotation=90)
            ax[im//Nx,im%Nx].set_yscale('log')
            ax[im//Nx,im%Nx].set_ylabel('concentration (M)')
            ax[im//Nx,im%Nx].grid(alpha=0.3)

        for im, mass in enumerate(list(mass2cmplxindex.keys())):
        
            im += len(mass2strandindex)

            # relevant concentrations
            cs = self.cs_cmplxs_equ[mass2cmplxindex[mass]]
            
            ax[im//Nx,im%Nx].set_title('mass: %s' %mass, fontsize=15)
            ax[im//Nx,im%Nx].scatter(np.arange(len(cs)), cs)
            ax[im//Nx,im%Nx].set_xticks(list(mass2cmplxprops_rel[mass].keys()), \
                    list(mass2cmplxprops_rel[mass].values()), rotation=90)
            ax[im//Nx,im%Nx].set_yscale('log')
            ax[im//Nx,im%Nx].set_ylabel('concentration (M)')
            ax[im//Nx,im%Nx].grid(alpha=0.3)

        if(filepath != ''):
            plt.savefig(filepath)

        plt.show()


    def save_equilibrium_concentrations(self, filepath):

        cmplxindexflat2id = {value:key for key,value in \
                             self.comp_cont.cmplxid2indexflat.items()}
    
        filestring = 'cmplx_id\t\tconcentration (M)\n'

        for i in self.comp_cont.strand_indices_sorted:
            conc_str = hf.build_scientific_notation(self.cs_strands_equ[i], 15)
            filestring += "%s\t\t%s\n" %(self.comp_cont.strands[i].id, conc_str)

        for i in self.comp_cont.cmplx_indices_flat_sorted:
            id = cmplxindexflat2id[i]
            conc = self.cs_cmplxs_equ[i]
            conc_str = hf.build_scientific_notation(conc, 15)
            filestring += "%s\t\t%s\n" %(id,conc_str)

        f = open(filepath, 'w')
        f.write(filestring)
        f.close()


    def save_equilibrium_concentrations_memoryfriendly(self, filepath):

        cs = []
        for i in self.comp_cont.strand_indices_sorted:
            cs.append(self.cs_strands_equ[i])

        for i in self.comp_cont.cmplx_indices_flat_sorted:
            cs.append(self.cs_cmplxs_equ[i])

        cs = np.asarray(cs)

        f = open(filepath, 'wb')
        pkl.dump(cs, f)
        f.close()


    def read_equilibrium_concentrations_memoryfriendly(self, filepath_ids, filepath_cs):

        if self.comp_cont.check_if_order_of_sorted_cmplxids_in_file_agrees_with_expected_order(filepath_ids):

            f = open(filepath_cs, 'rb')
            cs = pkl.load(f)
            f.close()
            
            self.cs_strands_equ = np.zeros(len(self.comp_cont.strand_indices_sorted))
            self.cs_cmplxs_equ = np.zeros(len(self.comp_cont.cmplx_indices_flat_sorted))

            ids = self.comp_cont._CompoundContainer__strandids_sorted + self.comp_cont._CompoundContainer__cmplxids_sorted

            for i in range(len(cs)):
                identity = ids[i]
                if identity[0] == 's':
                    index = self.comp_cont.strandid2index[identity]
                    self.cs_strands_equ[index] = cs[i]
                elif identity[0] == 'c':
                    index = self.comp_cont.cmplxid2indexflat[identity]
                    self.cs_cmplxs_equ[index] = cs[i]

        else:
            raise ValueError('strandids_cmplxids.txt is inconsistent with the provided CompoundContainer')


    def save_equilibrium_concentrations_productive_cmplxs(self, filepath):

        if(not hasattr(self, 'cs_prod')):
            self.identify_concentrations_productive_complexes()
        
        filestring = 'product type\t\tconcentration (M)\n'

        for subgroup_type in self.cs_prod.keys():
            conc_str = hf.build_scientific_notation(self.cs_prod[subgroup_type], 15)
            filestring += "%s\t\t%s\n" %(subgroup_type,conc_str)
        
        f = open(filepath, 'w')
        f.write(filestring)
        f.close()



    def save_equilibrium_concentrations_by_type(self, filepath):

        if(not hasattr(self, 'cs_bytype')):
            self.identify_concentrations_by_complex_type()

        filestring = 'complex type\t\tconcentration (M)\n'

        for subgroup_type in self.cs_bytype.keys():
            conc_str = hf.build_scientific_notation(self.cs_bytype[subgroup_type], 15)
            filestring += "%s\t\t%s\n" %(subgroup_type,conc_str)

        f = open(filepath, 'w')
        f.write(filestring)
        f.close()
