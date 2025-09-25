#!/bin/env python3

import numpy as np
import sys
sys.path.append('../src/')
import string

from utils import *
from metropolis_hastings import *

# parameters
## length genome
l_gen = 64
## characteristic length scales
L1 = 3
L2 = 5
## which length scales had been mutated, which where immutable
ls_motif__mut = np.array([4])
ls_motif__immut = np.array([1,2,3,5])
ls_motif_all = np.arange(1,L2+1,1)

# read the sampled genomes
dirpath = f'../outputs/lgen_{l_gen}/'
filename = f'genomes_and_entropies__lgen_{l_gen}__unif_{",".join([str(l) for l in ls_motif__immut])}__' \
           f'strongnonunif_{",".join([str(l) for l in ls_motif__mut])}__L1_{L1}__L2_{L2}.txt'
genomes, genome_keys, entropies = \
    read_genomes_and_entropies_from_file(filepath=dirpath+filename, l_gen=l_gen)

# create a histogram of the entropies of all sampled genomes
entropies_hist = {}
for entropies_loc in entropies[:,ls_motif__mut-1]:
    entropies_key = tuple(np.round(entropies_loc, 10))
    if not entropies_key in entropies_hist:
        entropies_hist[entropies_key] = 1
    else:
        entropies_hist[entropies_key] += 1

# sort the histogram by entropies, i.e., largest entropies first
entropies_hist = {key:entropies_hist[key] \
                  for key in sorted(entropies_hist.keys(), key=lambda el: [el])}
entropy_of_interest = list(entropies_hist.keys())[0]

# return the corresponding genome and its entropies
genome_indices = np.argwhere(np.all(np.round(entropies[:,ls_motif__mut-1],10) == entropy_of_interest, axis=1))[:,0]
genome_index = genome_indices[0]
genome_key = genome_keys[genome_index]
genome = genomes[genome_index]
entropies_out, freqs_out = compute_motif_entropies(genome, ls_motif_all)
print("genome key: ", genome_key)
print("genome: ", genome)
print("its entropies: ", entropies[genome_index])
