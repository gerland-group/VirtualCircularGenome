#!/bin/env python3

from functools import partial
from multiprocessing import Pool
from numba import njit
import numpy as np
import os
import tqdm

from utils import *

@njit
def mutate_single_letter(genome):
    letters = np.array([0,1,2,3])
    genome_out = np.copy(genome)
    pos_letter = np.random.randint(len(genome))
    letter_old = genome[pos_letter]
    letter_new = letters[letters != letter_old][np.random.randint(3)]
    genome_out[pos_letter] = letter_new
    return genome_out

@njit
def copy_and_paste(genome):
    i_start_1 = np.random.randint(len(genome))
    # l_word = np.random.randint(int(10*np.log(len(genome))/np.log(4)))
    l_word = np.random.randint(2*len(genome)//3)
    i_start_2 = np.random.randint((i_start_1+l_word)%len(genome), len(genome))
    # print("i_start_1: ", i_start_1)
    # print("i_start_2: ", i_start_2)
    # print("l_word: ", l_word)
    region_1 = extract_word(genome, i_start=i_start_1, l_word=l_word)
    region_2 = extract_word(genome, i_start=i_start_2, l_word=l_word)
    genome_out = np.copy(genome)
    if i_start_1+l_word < len(genome_out):
        genome_out[i_start_1:i_start_1+l_word] = region_2
    elif i_start_1+l_word >= len(genome_out):
        l_part_1 = len(genome)-i_start_1
        l_part_2 = l_word - l_part_1
        genome_out[i_start_1:] = region_2[0:l_part_1]
        genome_out[0:l_part_2] = region_2[l_part_1:l_word]
    if i_start_2+l_word < len(genome_out):
        genome_out[i_start_2:i_start_2+l_word] = region_1
    elif i_start_2+l_word >= len(genome_out):
        l_part_1 = len(genome)-i_start_2
        l_part_2 = l_word - l_part_1
        genome_out[i_start_2:] = region_1[0:l_part_1]
        genome_out[0:l_part_2] = region_1[l_part_1:l_word]
    return genome_out

@njit
def cut_and_paste(genome):
    l_word = np.random.randint(1, 2*int(np.log(len(genome))/np.log(4)))
    i_old = np.random.randint(len(genome)-l_word)
    i_new = np.random.randint(len(genome)-l_word)
    mask = np.array([False for _ in range(len(genome))])
    mask[i_old:i_old+l_word] = True
    genome_out = np.array([*genome[~mask][0:i_new], *genome[mask], *genome[~mask][i_new:]])
    return genome_out

@njit
def mutate(genome):
    r = np.random.random()
    ps = np.array([1., 10., 1.])
    pscum = np.cumsum(ps) / np.sum(ps)
    if r < pscum[0]:
        # single nucleotide mutation
        genome_out = mutate_single_letter(genome)
    elif r >= pscum[0] and r < pscum[1]:
        # cut and paste
        genome_out = cut_and_paste(genome)
    elif r >= pscum[1] and r < pscum[2]:
        # reshuffle
        pos = np.random.randint(len(genome))
        genome_out = np.array([*genome[pos:], *genome[0:pos]])
    return genome_out

@njit
def run_single_metropolis_hastings(genome_init, ls_motif, N_steps=int(1e5), \
                                   VERBOSE=False, tau=1e-3):
    # initial genome
    genome_old = genome_init
    
    # compute maximal entropy
    max_entropies = np.array([l if l <= np.log(2*len(genome_old))/np.log(4) \
                              else np.log(2*len(genome_old))/np.log(4) \
                              for l in ls_motif])
    max_entropy = np.sum(max_entropies)
    # initial entropy
    entropy_old = np.sum(compute_motif_entropies__wo_freqs(genome_old, ls_motif))
    if VERBOSE:
        print("entropy max: ", max_entropy)
    
    # start the loop
    n = 0
    entropies = [entropy_old]
    while (n < N_steps) and not np.isclose(entropy_old, max_entropy, rtol=1e-8):
        genome_new = mutate(genome_old)
        entropy_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif))
        if entropy_new >= entropy_old:
            # certain accept
            genome_old = genome_new
            entropy_old = entropy_new
        elif np.random.random() < np.exp((entropy_new - entropy_old)/tau):
            # conditional accept
            genome_old = genome_new
            entropy_old = entropy_new
        else:
            pass
        if (n%int(1e4) == 0) and VERBOSE:
            print("n: ", n, " entropy: ", entropy_old)
        n += 1
        entropies.append(entropy_old)
    genome = genome_old
    return genome, entropies

@njit
def run_single_metropolis_hastings__minimize_entropy_for_ls_motif_mut(\
        genome_init, ls_motif__immut, ls_motif__mut, \
        L1_target=None, L2_target=None, N_steps=int(1e5), VERBOSE=False, tau=1e-5):
    '''
    This function minimizes the entropy for the motifs of length ls_motif_mut
    while keeping the entropy of the motifs of length ls_motif_immut fixed 
    at their maximum possible value. 
    Performing this minimization can be useful, to generate a genome with a 
    (i) smaller L1 than the input genome and (ii) a larger L2 than the input genome.
    For this reason, it is possible to specify L1_target. If specified, MH terminates
    once L1_target is reached.
    '''
    
    # initial genome
    genome_old = genome_init
    
    # compute (maximal) entropy of immutable motif lengths
    # entropy_immut_max = np.sum(np.array([l if l <= np.log(2*len(genome_old))/np.log(4) \
    #                            else np.log(2*len(genome_old))/np.log(4) \
    #                            for l in ls_motif__immut]))
    entropy_immut_input = np.sum(compute_motif_entropies__wo_freqs(genome_old, ls_motif__immut))
    # assert np.isclose(entropy_immut_old, entropy_immut_max, rtol=1e-9)
    
    # compute entropy of mutable motif lengths
    entropy_mut_old = np.sum(compute_motif_entropies__wo_freqs(genome_old, ls_motif__mut))

    # start the loop
    n = 0
    entropies = [entropy_mut_old]
    if (L1_target == None) and (L2_target == None):
        while (n < N_steps):
            genome_new = mutate(genome_old)
            entropy_mut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__mut))
            entropy_immut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__immut))
            if np.isclose(entropy_immut_new, entropy_immut_input, rtol=1e-9):
                if entropy_mut_new <= entropy_mut_old:
                    # certain accept
                    genome_old = genome_new
                    entropy_mut_old = entropy_mut_new
                elif np.random.random() < np.exp((entropy_mut_old-entropy_mut_new)/tau):
                    # conditional accept
                    genome_old = genome_new
                    entropy_mut_old = entropy_mut_new
            if (n%int(1e4) == 0) and VERBOSE:
                print("n: ", n, " entropy mutable: ", entropy_mut_old, \
                      " L1: ", identify_L1__single_genome(genome_old), \
                      " L2: ", identify_L2__single_genome(genome_old))
            n += 1
            entropies.append(entropy_mut_old)
    elif (L1_target == None) and not (L2_target == None):
        while (n < N_steps) \
              and (identify_L2__single_genome(genome_old) != L2_target):
            genome_new = mutate(genome_old)
            entropy_mut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__mut))
            entropy_immut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__immut))
            if np.isclose(entropy_immut_new, entropy_immut_input, rtol=1e-9):
                if entropy_mut_new <= entropy_mut_old:
                    # certain accept
                    genome_old = genome_new
                    entropy_mut_old = entropy_mut_new
                elif np.random.random() < np.exp((entropy_mut_old-entropy_mut_new)/tau):
                    # conditional accept
                    genome_old = genome_new
                    entropy_mut_old = entropy_mut_new
            if (n%int(1e4) == 0) and VERBOSE:
                print("n: ", n, " entropy mutable: ", entropy_mut_old, \
                      " L1: ", identify_L1__single_genome(genome_old), \
                      " L2: ", identify_L2__single_genome(genome_old))
            n += 1
            entropies.append(entropy_mut_old)
    elif not (L1_target == None) and (L2_target == None):
        while (n < N_steps) \
              and (identify_L1__single_genome(genome_old) != L1_target):
            genome_new = mutate(genome_old)
            entropy_mut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__mut))
            entropy_immut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__immut))
            if np.isclose(entropy_immut_new, entropy_immut_input, rtol=1e-9):
                if entropy_mut_new <= entropy_mut_old:
                    # certain accept
                    genome_old = genome_new
                    entropy_mut_old = entropy_mut_new
                elif np.random.random() < np.exp((entropy_mut_old-entropy_mut_new)/tau):
                    # conditional accept
                    genome_old = genome_new
                    entropy_mut_old = entropy_mut_new
            if (n%int(1e4) == 0) and VERBOSE:
                print("n: ", n, " entropy mutable: ", entropy_mut_old, \
                      " L1: ", identify_L1__single_genome(genome_old), \
                      " L2: ", identify_L2__single_genome(genome_old))
            n += 1
            entropies.append(entropy_mut_old)
    elif not (L1_target == None) and not (L2_target == None):
        while (n < N_steps) \
              and ((identify_L1__single_genome(genome_old) != L1_target) \
                    or (identify_L2__single_genome(genome_old) != L2_target)):
            genome_new = mutate(genome_old)
            entropy_mut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__mut))
            entropy_immut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__immut))
            if np.isclose(entropy_immut_new, entropy_immut_input, rtol=1e-9):
                if entropy_mut_new <= entropy_mut_old:
                    # certain accept
                    genome_old = genome_new
                    entropy_mut_old = entropy_mut_new
                elif np.random.random() < np.exp((entropy_mut_old-entropy_mut_new)/tau):
                    # conditional accept
                    genome_old = genome_new
                    entropy_mut_old = entropy_mut_new
            if (n%int(1e4) == 0) and VERBOSE:
                print("n: ", n, " entropy mutable: ", entropy_mut_old, \
                      " L1: ", identify_L1__single_genome(genome_old), \
                      " L2: ", identify_L2__single_genome(genome_old))
            n += 1
            entropies.append(entropy_mut_old)
    genome = genome_old
    return genome, entropies

@njit
def run_single_metropolis_hastings__reach_target_entropy_for_ls_motif_mut(\
        genome_init, ls_motif__immut, ls_motif__mut, target_entropy_mut, \
        N_steps=int(1e5), VERBOSE=False, tau=1):
    '''
    This function mutates the genome until the target entropy is reached for the 
    motifs with length ls_motif__mut (mutable motifs). For all other motifs, the
    entropy is constrained to be fixed to its maximum value.
    '''
    assert len(target_entropy_mut) == len(ls_motif__mut)
    # initial genome
    genome_old = genome_init
    
    # compute entropy of immutable motif lengths
    # entropy_immut_max = np.sum(np.array([l if l <= np.log(2*len(genome_old))/np.log(4) \
    #                            else np.log(2*len(genome_old))/np.log(4) \
    #                            for l in ls_motif__immut]))
    entropy_immut_input = np.sum(compute_motif_entropies__wo_freqs(genome_old, ls_motif__immut))
    # assert np.isclose(entropy_immut_old, entropy_immut_max, rtol=1e-9)
    
    # compute entropy of mutable motif lengths
    entropy_mut_old = compute_motif_entropies__wo_freqs(genome_old, ls_motif__mut)

    if VERBOSE:
        print("target entropy mutable: ", target_entropy_mut)

    # start the loop
    n = 0
    entropies = [entropy_mut_old]
    while (n < N_steps) and not np.all(np.isclose(entropy_mut_old, target_entropy_mut)):
        genome_new = mutate(genome_old)
        entropy_mut_new = compute_motif_entropies__wo_freqs(genome_new, ls_motif__mut)
        entropy_immut_new = np.sum(compute_motif_entropies__wo_freqs(genome_new, ls_motif__immut))
        if np.isclose(entropy_immut_new, entropy_immut_input, rtol=1e-9):
            delta = np.sum((entropy_mut_new-target_entropy_mut)**2) \
                    - np.sum((entropy_mut_old-target_entropy_mut)**2)
            delta *= 1e13
            if delta < 0:
                # certain accept
                genome_old = genome_new
                entropy_mut_old = entropy_mut_new
            elif np.random.random() < np.exp(-np.sqrt(delta)/tau):
                # conditional accept
                genome_old = genome_new
                entropy_mut_old = entropy_mut_new
        if (n%int(1e4) == 0) and VERBOSE:
            print("n: ", n, " entropy mutable: ", entropy_mut_old)
        n += 1
        entropies.append(entropy_mut_old)
    genome = genome_old
    return genome, entropies

def run_multiple_metropolis_hastings(l_gen, ls_motif, N_steps, N_runs, tau=1e-3):
    
    # p = Pool(int(2/3*os.cpu_count()))
    p = Pool(os.cpu_count())
    genomes_init = [np.random.randint(0,4,size=l_gen) for _ in range(N_runs)]
    entropies_out = []
    genomes_out = [] 
    run_single_eff = partial(run_single_metropolis_hastings, ls_motif=ls_motif, N_steps=N_steps, \
                             VERBOSE=False, tau=tau)
    for out in tqdm.tqdm(p.imap(run_single_eff, genomes_init), total=N_runs):
        genome_out, entropies_loc = out
        genomes_out.append(genome_out)
        entropies_out.append(entropies_loc[-1])
    p.close()
    return genomes_out, entropies_out

def run_multiple_metropolis_hastings__minimize_entropy_for_ls_motif_mut(\
    genomes_init, ls_motif__immut, ls_motif__mut, L1_target, L2_target, \
    N_steps, tau=1e-5, OUTPUT_TRAJECTORY=False):

    p = Pool(os.cpu_count())
    entropies_out = []
    genomes_out = []
    run_single_eff = partial(run_single_metropolis_hastings__minimize_entropy_for_ls_motif_mut, \
        ls_motif__immut=ls_motif__immut, ls_motif__mut=ls_motif__mut, \
        L1_target=L1_target, L2_target=L2_target, N_steps=N_steps, tau=tau)
    for out in tqdm.tqdm(p.imap(run_single_eff, genomes_init), total=len(genomes_init)):
        genome_out, entropies_loc = out
        genomes_out.append(genome_out)
        if not OUTPUT_TRAJECTORY:
            entropies_out.append(entropies_loc[-1])
        elif OUTPUT_TRAJECTORY:
            # output 1000 data points along the time-evolution of the entorpies
            entropies_out.append(entropies_loc[::int(N_steps)//1000]) 
    p.close()
    return genomes_out, entropies_out
