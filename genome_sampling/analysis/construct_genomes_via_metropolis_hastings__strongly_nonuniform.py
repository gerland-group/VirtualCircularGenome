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
ls_motif_in = np.array([1,2,3,4])
## characteristic length scales of ingoing and outgoing genome
L1_target_in = 3
L1_target_out = 3
L2_target_in = 4
L2_target_out = 5
## length scales on which the motif entropy is supposed to be reduced (ls_motif__mut)
ls_motif__mut = np.array([4])
## length scales on which the motif entropy is supposed to be unchanged (ls_motif__immut)
ls_motif__immut = np.array([1,2,3,5])

# read the genomes
dirpath = f'../outputs/lgen_{l_gen}/'
filename = f'genomes_and_entropies__lgen_{l_gen}__'\
           f'unif_{",".join([str(l) for l in ls_motif_in])}__'\
           f'L1_{L1_target_in}__L2_{L2_target_in}.txt'
genomes_in, _, entropies_in = \
    read_genomes_and_entropies_from_file(filepath=dirpath+filename, l_gen=64)

genomes, entropies_evol = \
    run_multiple_metropolis_hastings__minimize_entropy_for_ls_motif_mut(\
    genomes_in, ls_motif__immut=ls_motif__immut, ls_motif__mut=ls_motif__mut, \
    L1_target=None, L2_target=None, N_steps=int(3e6), tau=1e-5, OUTPUT_TRAJECTORY=True)
# note: setting L1_target and L2_target = None is essential, as this avoids termination
# of the genome search based on L1 and L2
entropies_evol = np.asarray(entropies_evol)

genomes_rel = []
for genome in genomes:
    L1 = identify_L1__single_genome(genome)
    L2 = identify_L2__single_genome(genome)
    if (L1 == L1_target_out) and (L2 == L2_target_out):
        genomes_rel.append(genome)

ls_motif_all = np.arange(1, L2_target_out+1, 1)
entropies_rel = np.zeros((len(genomes_rel), len(ls_motif_all)))
for i, genome in enumerate(genomes_rel):
    entropies_rel[i] = compute_motif_entropies__wo_freqs(genome, ls_motif=ls_motif_all)

# write data to file
write_genomes_and_entropies_to_file(genomes_rel, entropies_rel, l_gen=l_gen, ls_motif=ls_motif_all, \
    dirpath=f'../outputs/lgen_{l_gen}', \
    filename=f'genomes_and_entropies__lgen_{l_gen}__unif_{",".join([str(l) for l in ls_motif__immut])}__' \
             f'strongnonunif_{",".join([str(l) for l in ls_motif__mut])}__L1_{L1_target_out}__L2_{L2_target_out}.txt')
# save the evolution of entropies along the Metropolis-Hastings simulation
np.savetxt(f'../outputs/lgen_{l_gen}/entropies_timeevol__' \
           f'lgen_{l_gen}__unif_{",".join([str(l) for l in ls_motif__immut])}__' \
           f'strongnonunif_{",".join([str(l) for l in ls_motif__mut])}__L1_{L1_target_out}__L2_{L2_target_out}.txt', \
           entropies_evol)
