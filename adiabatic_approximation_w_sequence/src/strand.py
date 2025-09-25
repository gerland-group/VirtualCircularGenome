#!/bin/env python3

from global_variables import *
import helper_functions as hf

class Strand:

    def __init__(self, seq, INV=False, numb=None, conc=None):

        if(INV):
            self.seq_5t3 = seq[::-1]
        else:
            self.seq_5t3 = seq

        self.l = len(self.seq_5t3)
        self.id = "s_%s" %int(hf.convert_AGCT_to_1234(self.seq_5t3), base=(alphabet+1))
        self.numb = numb # copy number
        self.conc = conc # in units M
        
    
    def __str__(self):
        
        return self.seq_5t3


    def determine_type(self):

        self_comps = ['AT','TA','CG','GC']
        if(self.seq_5t3[0:2] in self_comps and self.seq_5t3[-2:] in self_comps):
            self.type = '2c'
        elif(self.seq_5t3[0:2] in self_comps or self.seq_5t3[-2:] in self_comps):
            self.type = '1c'
        else:
            self.type = '0c'
