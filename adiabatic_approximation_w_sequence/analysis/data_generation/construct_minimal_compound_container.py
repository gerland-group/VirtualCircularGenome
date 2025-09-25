#!/bin/env python3

import sys
sys.path.append('../../src/')
import pickle as pkl
import pandas as pd

from global_variables import *
from genome import *
from strand import *
from helix import *
from complex import *
from compound_container import *

def print_parameter_set(parameter_set):
    Lgen, L1, L2, bias, oligomer_lengths = parameter_set
    print("L_G: %d, L_1: %d, L_2: %d, bias: %s, oligomer_lenghts: %s" %(Lgen,L1,L2,bias,oligomer_lengths))

def construct_compound_container(parameter_set):
    # extract parameters
    Lgen, L1, L2, bias, ls = parameter_set
    with open('../../inputs/Lgen_%s__L1_%s__L2_%s__%s/genome_key.txt' %(Lgen, L1, L2, bias), 'r') as file:
        filestring = file.read()
    genome_keys = [el for el in filestring.split('\n') if el != '']
    assert len(genome_keys) == 1
    genome_key = genome_keys[0]
    print("genome_key: ", genome_key)

    # construct the genome
    gen = Genome(genome_key, alphabet, 2*Lgen, lmax=2*np.max(ls))
    gen.list_included_words()

    # construct the compound container
    container = CompoundContainer(\
        '../../inputs/Lgen_%s__L1_%s__L2_%s__%s/complexes__ls_%s.txt' \
        %(Lgen, L1, L2, bias, '_'.join(map(str, ls))), \
        read_copy_numbers=False, is_truncated=True, truncation_type='mismatch')
    container.group_strands_by_length_and_terminalletters()
    print("constructing duplexes")
    container.list_all_duplexes()
    print("constructing triplexes")
    container.list_all_nplus1_plexes_no_mismatches(2)
    #print("constructing tetraplexes")
    #container.list_all_nplus1_plexes_no_mismatches(3, VERBOSE=True)
    container.list_productive_complexes(genome=gen)

    # set up the energy calculator (to compute the binding affinities)
    container.ec = EnergyCalculator('../../inputs/energy_parameters.txt')
    container.compute_dissociation_constants_all_cmplxs()
    container.compute_masses_all_cmplxs()
    container.compute_length_of_hybridization_sites_all_cmplxs()
    container.compute_number_of_matches_and_mismatches_all_cmplxs()
    container.construct_cmplx_properties_flat()
    container.create_map_cmplxid_to_cmplxindexflat()
    container.create_map_strandindex_to_cmplxindex_flat()

    # change from numba dictionary to regular dictionary such that object can be pickled
    container.strand_indices_flat = list(container.strand_indices_flat)

    # overwrite dictionary of complexes to save storage space
    container.cmplxs = {}

    # write the complex container
    g = open('../../inputs/Lgen_%s__L1_%s__L2_%s__%s/minimal_container__ls_%s.pkl' \
            %(Lgen, L1, L2, bias, '_'.join(map(str, ls))), 'wb')
    pkl.dump(container, g)
    g.close()

# run single compound container construction from input arguments
Lgen = int(sys.argv[1])
L1 = int(sys.argv[2])
L2 = int(sys.argv[3])
bias = str(sys.argv[4])
ls = str(sys.argv[5])
assert ls[0] == '[' and ls[-1] == ']'
ls = [int(el) for el in ls[1:-1].split(',')]
parameter_set = (Lgen,L1,L2,bias,ls)
construct_compound_container(parameter_set)
