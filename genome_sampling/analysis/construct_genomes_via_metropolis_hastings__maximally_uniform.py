#!/bin/env python3

import numpy as np
import sys
sys.path.append('../src/')
import string

from utils import *
from metropolis_hastings import run_multiple_metropolis_hastings, run_single_metropolis_hastings

# parameters
## length genome
l_gen = 64
## length scales of motifs for which the entropy is maximized
ls_motif = np.array([1,2,3,4])
L1_target = 3
L2_target = 4
## number of metropolis-hastings simulations
N_runs = 1000

# create list of genomes (and corresponding entropies) that maximize the entropy
# for the lenght-scales specified in ls_motif via Metropolis-Hastings
genomes_out, entropies_out = run_multiple_metropolis_hastings(l_gen=l_gen, \
    ls_motif=ls_motif, N_steps=int(1e6), N_runs=int(N_runs), tau=1e-5)

# extract the genomes that did attain (i) the required entropy and (ii) the desired
# target length
entropy_max = np.sum(np.array([l if l <= np.log(2*l_gen)/np.log(4) \
                               else np.log(2*l_gen)/np.log(4) \
                               for l in ls_motif]))
genomes_rel = []
for genome in genomes_out:
    entropy = np.sum(compute_motif_entropies__wo_freqs(genome, ls_motif=ls_motif))
    L1 = identify_L1__single_genome(genome)
    L2 = identify_L2__single_genome(genome)
    if np.all(np.isclose(entropy, entropy_max)) and L2 == L2_target:
        genomes_rel.append(genome)
        assert L1 == L1_target

# compute the entropy of the unconstrained motifs
ls_motif_all = np.arange(1,np.max(ls_motif)+1,1)
entropies_rel = np.zeros((len(genomes_rel),len(ls_motif_all)))
for i, genome in enumerate(genomes_rel):
    entropies_rel[i] = compute_motif_entropies__wo_freqs(genome, ls_motif=ls_motif_all)

# write data to file
write_genomes_and_entropies_to_file(\
    genomes_rel, entropies_rel, l_gen=l_gen, ls_motif=ls_motif, \
    dirpath=f'../outputs/lgen_{l_gen}/', \
    filename=f'genomes_and_entropies__lgen_{l_gen}__unif_{",".join([str(l) for l in ls_motif])}' \
             +f'__L1_{L1_target}__L2_{L2_target}.txt'
)
