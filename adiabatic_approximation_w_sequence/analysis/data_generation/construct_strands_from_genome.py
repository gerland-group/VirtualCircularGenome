#!/bin/env python3

import pandas as pd
import sys
sys.path.append('../../src/')
import yaml

from genome import *

class Strands:

    def __init__(self, ls, gen: Genome):

        self.ls = ls # length of the strands that are supposed to be included
        self.gen = gen
        for l in self.ls:
            self.gen.extend_list_included_words(l)
        self.construct_list_of_all_strands()

    def construct_list_of_all_strands(self):

        self.seqs = []
        for l in self.ls:
            for seq, number in self.gen.included_words[l].items():
                self.seqs.append(seq)

    def write_to_file(self, filepath):

        local_filestring = "N=1\n3'%s 5'\n5'%s 3'\n\n"

        filestring = ""
        for seq in self.seqs:
            filestring += (local_filestring %('.'*len(seq), seq))
        filestring = filestring[0:-1]

        f = open(filepath, 'w')
        f.write(filestring)
        f.close()

def print_parameter_set(parameter_set):
    Lgen, L1, L2, bias, oligomer_lengths = parameter_set
    print("L_G: %d, L_1: %d, L_2: %d, bias: %s, oligomer_lenghts: %s" %(Lgen,L1,L2,bias,oligomer_lengths))

def construct_genome_and_strands(parameter_set):
    # extract parameters
    Lgen, L1, L2, bias, oligomer_lengths = parameter_set

    # identify genome key
    with open('../../inputs/Lgen_%s__L1_%s__L2_%s__%s/genome_key.txt' %(Lgen, L1, L2, bias), 'r') as file:
        filestring = file.read()
    genome_keys = [el for el in filestring.split('\n') if el != '']
    assert len(genome_keys) == 1
    genome_key = genome_keys[0]

    # construct genome
    genome= Genome(genome_key, 4, 2*Lgen, lmax=1)
    genome.list_included_words()

    # construct strands
    strnds = Strands(oligomer_lengths, gen=genome)
    strnds.write_to_file(filepath='../../inputs/Lgen_%s__L1_%s__L2_%s__%s/complexes__ls_%s.txt' \
                         %(Lgen, L1, L2, bias, '_'.join([str(el) for el in oligomer_lengths])))

with open('./params.yaml', 'r') as file:
    params = yaml.safe_load(file)

l_gen = params['l_gen']
L1 = params['L1']
L2 = params['L2']
bias = params['bias']
ls_oligo_V = params['ls_oligo_V']

parameter_sets = [(l_gen,L1,L2,bias,[1,l_oligo_V]) for l_oligo_V in ls_oligo_V]
for parameter_set in parameter_sets:
    print_parameter_set(parameter_set)
    construct_genome_and_strands(parameter_set)    
