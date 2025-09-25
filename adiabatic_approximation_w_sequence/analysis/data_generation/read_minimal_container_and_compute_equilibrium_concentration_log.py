#!/bin/env python3

# import common modules
import numba
import os
import pickle as pkl
import sys
import pandas as pd
sys.path.append('../../src/')
import yaml

# import own modules
from global_variables import *
from genome import *
from strand import *
from helix import *
from complex import *
from compound_container import *
from energy_calculator import *
from concentration_computer import *
import helper_functions as hf

print("read genome parameters")
with open('./params.yaml', 'r') as file:
    params = yaml.safe_load(file)

genome_length = params['l_gen']
L1 = params['L1']
L2 = params['L2']
bias = params['bias']
l_oligo_V = int(sys.argv[1])
ls = [1,l_oligo_V]

print("l_oligo_V: ", l_oligo_V)

print("specify concentration of VCG oligos")
cFtot = 1e-4
cVtot = float(sys.argv[2])

with open('../../inputs/Lgen_%s__L1_%s__L2_%s__%s/genome_key.txt' %(genome_length, L1, L2, bias), 'r') as file:
    filestring = file.read()
genome_keys = [el for el in filestring.split('\n') if el != '']
assert len(genome_keys) == 1
genome_key = genome_keys[0]

cs_tot__array = [cFtot / 4**ls[0], cVtot / (genome_length*2)]
cs_tot__dict = dict(zip(ls, cs_tot__array))

print("construct genome")
gen = Genome(genome_key, alphabet, 2*genome_length, lmax=2*np.max(ls))
gen.list_included_words()

print("read compound container")
f = open('../../inputs/Lgen_%s__L1_%s__L2_%s__%s/minimal_container__ls_%s.pkl' \
          %(genome_length, L1, L2, bias, '_'.join(map(str, ls))), 'rb')
container: CompoundContainer = pkl.load(f)
f.close()

# convert regular list to numba-typed list
container.strand_indices_flat = numba.typed.List(container.strand_indices_flat)

# construct and save strandids and cmplxids sorted by properties to private variable
container.construct_sorted_strandids_and_sorted_cmplxids(make_private=True)

print("set strand concentrations in the container")
container.set_initial_strand_concentration_monomer_hexamer(cs_tot__dict)

print("define concentration computer, i. e. instance that handles the computation " \
      "of the equilibrium concentrations")
computer = ConcentrationComputer(container)

print("compute equilibrium")
computer.compute_equilibrium_concentration_log()
print("identify_concentrations_productive_complexes")
computer.identify_concentrations_productive_complexes()

# create directory if not existing
dirpath = '../../outputs/data/Lgen_%s__L1_%s__L2_%s__%s/ls_%s/' \
          %(genome_length, L1, L2, bias, '_'.join(map(str, ls)))
if not os.path.exists(dirpath):
    print(f"creating directory {dirpath}")
    os.makedirs(dirpath)

print("save productive concentrations")
computer.save_equilibrium_concentrations_productive_cmplxs(\
    dirpath + \
    f'concentrations_productive__cstot_{cFtot:1.3e}_{cVtot:1.3e}.txt')

## saving the concentrations of all complexes
## possible, but here not used to limit the memory size of the output
# if(computer.comp_cont.check_if_order_of_sorted_cmplxids_agrees_with_expected_order()):
#     print("save equilibrium concentrations")
#     computer.save_equilibrium_concentrations_memoryfriendly(\
#         dirpath + \
#         f'concentrations__cstot_{cFtot:1.3e}_{cVtot:1.3e}.pkl')
# else:
#     print("ERROR: equilibrium concentrations could not be stored due to inconsistent "\
#             "order of strandids or cmplxids")

