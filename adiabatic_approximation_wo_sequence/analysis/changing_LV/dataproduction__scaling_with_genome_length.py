#!/bin/env python3

import sys
sys.path.append('../../src/')
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.style.use('seaborn-v0_8-colorblind')
params = {'legend.fontsize': 'small'}
pylab.rcParams.update(params)
from scipy.optimize import curve_fit

from ComplexConstructor import *
from ConcentrationComputer import *
from src__scaling_effective_association_constants import f_lin, f_exp, DataSet
from src__influence_of_LV import AnalyticPredictionNew

# compute Ls based on Lg
def compute_unique_subsequence_length(Lg):
    Ls = np.ceil(np.log(Lg)/(np.log(4)) + np.log(2)/np.log(4))
    return Ls

def compute_characteristic_oligomer_length__single_Lg(Lg, gamma):
    Ls = compute_unique_subsequence_length(Lg)
    
    # analyze the scaling of the effective association constants
    ds = DataSet(gamma=gamma, l_unique=Ls, Ls_vcg=np.arange(Ls, 5*Ls,1,dtype=int), comb_vcg=2*Lg)
    ds.construct_effective_association_constants()
    ds.perform_curve_fit()
    ds.check_validity_of_approximation()
    ds.create_dictionary_of_fitted_parameters()

    # predict the characteristic length scale
    ap = AnalyticPredictionNew(params=ds.params, Ls=np.arange(Ls, 5*Ls,1,dtype=int))
    ap.compute_characteristic_lengthscale()
    return ap.L_c

def compute_characteristic_oligomer_length__multi_Lg(Lgs, gamma):
    Lcs = np.zeros(len(Lgs))
    for i, Lg in enumerate(Lgs):
        Lcs[i] = compute_characteristic_oligomer_length__single_Lg(Lg, gamma)
    return Lcs

Lgs = np.logspace(1, 3, 500)
Lss_disc = compute_unique_subsequence_length(Lgs)
Lss_cont = np.log(Lgs)/(np.log(4)) + np.log(2)/np.log(4)

print("computing data for gamma = -1.25 kT")
Lcs_13 = compute_characteristic_oligomer_length__multi_Lg(Lgs, gamma=-1.25)
print("computing data for gamma = -1.875 kT")
Lcs_18 = compute_characteristic_oligomer_length__multi_Lg(Lgs, gamma=-1.875)
print("computing data for gamma = -2.50 kT")
Lcs_25 = compute_characteristic_oligomer_length__multi_Lg(Lgs, gamma=-2.5)

np.savetxt("../../outputs/changing_LV/data/Lg.txt", Lgs)
np.savetxt("../../outputs/changing_LV/data/Ls_cont.txt", Lss_cont)
np.savetxt("../../outputs/changing_LV/data/Ls_disc.txt", Lss_disc)
np.savetxt("../../outputs/changing_LV/data/Lc_gamma_1.25.txt", Lcs_13)
np.savetxt("../../outputs/changing_LV/data/Lc_gamma_1.88.txt", Lcs_18)
np.savetxt("../../outputs/changing_LV/data/Lc_gamma_2.50.txt", Lcs_25)

# plot
f, axs = plt.subplots(1,2,figsize=(2*4.5,3.2), constrained_layout=True)

axs[0].plot(Lgs, Lss_disc, label="discrete")
axs[0].plot(Lgs, Lss_cont, label="continuous")
axs[0].set_xlabel('genome length $L_G$ (nt)')
axs[0].set_ylabel('unique subsequence length $L_S$ (nt)')
axs[0].set_xscale('log')
axs[0].legend()

axs[1].plot(Lgs, Lcs_13, label=r'$\gamma = -1.25\,k_\mathrm{B} \mathrm{T}$')
axs[1].plot(Lgs, Lcs_18, label=r'$\gamma = -1.88\,k_\mathrm{B} \mathrm{T}$')
axs[1].plot(Lgs, Lcs_25, label=r'$\gamma = -2.50\,k_\mathrm{B} \mathrm{T}$')
axs[1].set_xlabel('genome length $L_G$ (nt)')
axs[1].set_ylabel('characteristic oligomer length $L$ (nt)')
axs[1].set_xscale('log')
axs[1].legend()
