#!/bin/env python3

import sys
sys.path.append('../../../src/')
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.style.use('seaborn-v0_8-colorblind')
# params = {'legend.fontsize': 'small'}
# pylab.rcParams.update(params)
from scipy.optimize import curve_fit

from ComplexConstructor import *
from ConcentrationComputer import *


def f_lin(x, y0, m):
    return m*x-y0

def f_exp(x, y0, xi):
    return y0*np.exp(x/xi)

def f_quadexp(x, y0, xi):
    return y0*x**2*np.exp(x/xi)

def f_linexp(x, y0, xi):
    return y0*x*np.exp(x/xi)


class DataPoint:

    def __init__(self, L_vcg, gamma, l_unique=3, comb_vcg=32):

        self.cmplxs = ComplexConstructor(l_unique=l_unique, alphabet=4, \
                        L_vcg=L_vcg, L_vcg_min=L_vcg, L_vcg_max=L_vcg, \
                        Lmax_stock=1, comb_vcg=comb_vcg, gamma_2m=gamma, gamma_d=gamma/2)
        
        self.cncs = ConcentrationComputer(cmplxs=self.cmplxs, \
                        c0_vcg_all_oligos=0., \
                        Lambda_vcg=np.inf, \
                        c0_stock_all_oligos=0., \
                        Lambda_stock=np.inf)
        
        self.cncs.compute_ratio_concentration_strand_to_reference_strand()
        self.cncs.compute_complex_weights()
        self.cncs.compute_weights_productive_cvflvs()
        self.cncs.compute_coefficients_all_cvflvs_nonuc()
        self.cncs.compute_coefficients_errorfree_cvflvs_nonuc()
        self.cncs.compute_coefficients_error_cvflvs_nonuc()


class DataSet:

    def __init__(self, gamma, Ls_vcg=np.array([5,6,7,8,9,10,11,12,13,14,15,16]), \
                 l_unique=3, comb_vcg=32):

        self.l_unique = l_unique
        self.comb_vcg = comb_vcg
        self.Ls_vcg = Ls_vcg
        self.Ls_vcg_cont = np.linspace(self.Ls_vcg[0], self.Ls_vcg[-1], 100)
        self.gamma = gamma
        
        # arrays to store effective association constants
        self.Ka_FF_all = []
        self.Ka_FF_corr = []
        self.Ka_FV_all = []
        self.Ka_FV_corr = []
        self.Ka_VV_all = []
        self.Ka_VV_corr = []
        self.Ka_VV_err = []

    
    def construct_effective_association_constants(self):

        for L_vcg in self.Ls_vcg:
            dp = DataPoint(L_vcg=L_vcg, gamma=self.gamma, l_unique=self.l_unique, \
                           comb_vcg=self.comb_vcg)
            
            self.Ka_FF_all.append(dp.cncs.beta_all_nonuc)
            self.Ka_FF_corr.append(dp.cncs.beta_corr_nonuc)
            
            self.Ka_FV_all.append(dp.cncs.gamma_all_nonuc)
            self.Ka_FV_corr.append(dp.cncs.gamma_corr_nonuc)
            
            self.Ka_VV_all.append(dp.cncs.delta_all_nonuc)
            self.Ka_VV_corr.append(dp.cncs.delta_corr_nonuc)
            self.Ka_VV_err.append(dp.cncs.delta_err_nonuc)

        
        for var in ['Ka_FF_all', 'Ka_FF_corr', 'Ka_FV_all', 'Ka_FV_corr', \
                    'Ka_VV_all', 'Ka_VV_corr', 'Ka_VV_err']:
            exec(f'self.{var} = np.asarray(self.{var})')
    
    def perform_curve_fit(self):
        self.perform_curve_fit_FF()
        self.perform_curve_fit_FV()
        self.perform_curve_fit_VV()

    def perform_curve_fit_FF(self):
        out = curve_fit(f_lin, self.Ls_vcg, self.Ka_FF_all)[0]
        self.Ka0_FF_all = out[0]
        self.Lambda_FF_all = out[0]/out[1]
    
    def perform_curve_fit_FV(self):
        
        xi = (self.Ls_vcg[-1] - self.Ls_vcg[0])/np.log(self.Ka_FV_all[-1]/self.Ka_FV_all[0])
        y0 = self.Ka_FV_all[-1]*np.exp(-self.Ls_vcg[-1]/xi)
        out = curve_fit(f_exp, self.Ls_vcg, self.Ka_FV_all, p0=[y0,xi])[0]
        self.Ka0_FV_all = out[0]
        self.Lambda_FV_all = out[1]

        xi = (self.Ls_vcg[-1] - self.Ls_vcg[0])/np.log(self.Ka_FV_corr[-1]/self.Ka_FV_corr[0])
        y0 = self.Ka_FV_corr[-1]*np.exp(-self.Ls_vcg[-1]/xi)
        out = curve_fit(f_exp, self.Ls_vcg, self.Ka_FV_corr, p0=[y0,xi])[0]
        self.Ka0_FV_corr = out[0]
        self.Lambda_FV_corr = out[1]

    def perform_curve_fit_VV(self):

        xi = (self.Ls_vcg[-1] - self.Ls_vcg[0])/np.log(self.Ka_VV_all[-1]/(self.Ka_VV_all[0]))
        y0 = self.Ka_VV_all[0]*np.exp(-self.Ls_vcg[0]/xi)
        out = curve_fit(f_exp, self.Ls_vcg, self.Ka_VV_all, p0=[y0,xi])[0]
        # out = curve_fit(f_linexp, self.Ls_vcg, self.Ka_VV_all, p0=[y0,xi])[0]
        self.Ka0_VV_all = out[0]
        self.Lambda_VV_all = out[1]

        xi = (self.Ls_vcg[-1] - self.Ls_vcg[0])/np.log(self.Ka_VV_corr[-1]*self.Ls_vcg[0]/(self.Ka_VV_corr[0]*self.Ls_vcg[-1]))
        y0 = self.Ka_VV_corr[-1]*np.exp(-self.Ls_vcg[-1]/xi)/self.Ls_vcg[-1]
        # out = curve_fit(f_exp, self.Ls_vcg, self.Ka_VV_corr, p0=[y0,xi])[0]
        out = curve_fit(f_linexp, self.Ls_vcg, self.Ka_VV_corr, p0=[y0,xi])[0]
        self.Ka0_VV_corr = out[0]
        self.Lambda_VV_corr = out[1]

        xi = (self.Ls_vcg[-1] - self.Ls_vcg[0])/np.log(self.Ka_VV_err[-1]/(self.Ka_VV_err[0]))
        y0 = self.Ka_VV_err[0]*np.exp(-self.Ls_vcg[0]/xi)
        out = curve_fit(f_exp, self.Ls_vcg, self.Ka_VV_err, p0=[y0,xi])[0]
        # out = curve_fit(f_linexp, self.Ls_vcg, self.Ka_VV_all, p0=[y0,xi])[0]
        self.Ka0_VV_err = out[0]
        self.Lambda_VV_err = out[1]


    def print_fitted_parameters(self):

        print(f"Ka0_FF_all: {self.Ka0_FF_all:1.3e}")
        print(f"Lambda_FF_all: {self.Lambda_FF_all:1.3e}")
        
        print(f"Ka0_FV_all: {self.Ka0_FV_all:1.3e}")
        print(f"Lambda_FV_all: {self.Lambda_FV_all:1.3e}")
        
        print(f"Ka0_FV_corr: {self.Ka0_FV_corr:1.3e}")
        print(f"Lambda_FV_corr: {self.Lambda_FV_corr:1.3e}")

        print(f"Ka0_VV_all: {self.Ka0_VV_all:1.3e}")
        print(f"Lambda_VV_all: {self.Lambda_VV_all:1.3e}")
        
        print(f"Ka0_VV_corr: {self.Ka0_VV_corr:1.3e}")
        print(f"Lambda_VV_corr: {self.Lambda_VV_corr:1.3e}")

        print(f"Ka0_VV_err: {self.Ka0_VV_err:1.3e}")
        print(f"Lambda_VV_err: {self.Lambda_VV_err:1.3e}")

    
    def create_dictionary_of_fitted_parameters(self):

        self.params = {
            'Ka0_FF_all':self.Ka0_FF_all, \
            'Lambda_FF_all':self.Lambda_FF_all, \
            'Ka0_FV_all':self.Ka0_FV_all, \
            'Lambda_FV_all':self.Lambda_FV_all, \
            'Ka0_FV_corr':self.Ka0_FV_corr, \
            'Lambda_FV_corr':self.Lambda_FV_corr, \
            'Ka0_VV_all':self.Ka0_VV_all, \
            'Lambda_VV_all':self.Lambda_VV_all, \
            'Ka0_VV_corr':self.Ka0_VV_corr, \
            'Lambda_VV_corr':self.Lambda_VV_corr, \
            'Ka0_VV_err':self.Ka0_VV_err, \
            'Lambda_VV_err':self.Lambda_VV_err
            }
        
    def check_validity_of_approximation(self):
        part1 = self.Ka_FF_all*self.Ka_VV_corr/(self.Ka_VV_all*self.Ka_FV_corr - self.Ka_VV_corr*self.Ka_FV_all)
        part2_num = np.sqrt( self.Ls_vcg**2 * self.Ka_FF_all**2 * self.Ka_VV_corr**2 \
                            + self.Ls_vcg * self.Ka_FF_all * self.Ka_FV_corr * (self.Ka_VV_all*self.Ka_FV_corr - self.Ka_VV_corr*self.Ka_FV_all))
        part2_denom = self.Ls_vcg * (self.Ka_VV_all*self.Ka_FV_corr - self.Ka_VV_corr*self.Ka_FV_all)
        ropt = part1 + part2_num/part2_denom
        self.small_correction_in_root = self.Ls_vcg * self.Ka_FF_all**2 * self.Ka_VV_corr**2 \
                                  / (self.Ka_FF_all * self.Ka_FV_all**2 * (self.Ka_VV_all - self.Ka_VV_corr))
        self.small_correction_outside_root = self.Ka_FF_all*self.Ka_VV_corr/(self.Ka_VV_all*self.Ka_FV_corr - self.Ka_VV_corr*self.Ka_FV_all) / ropt
