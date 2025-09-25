#!/bin/env python3

from numba import njit
import numpy as np

# nomenclature:
# 0: A, 1: C, 2: G, 3: T

def dict_to_array(dictionary):
    return np.asarray(list(dictionary.keys())), np.asarray(list(dictionary.values()))

def array_to_hist(array):
    dictionary = {}
    for el in array:
        if not el in dictionary:
            dictionary[el] = 1
        else:
            dictionary[el] += 1
    dictionary = {key:dictionary[key] for key in sorted(list(dictionary.keys()))}
    return dictionary

@njit
def construct_complementary_genome(genome):
    return 3-genome[::-1]

@njit
def extract_word(genome: np.ndarray, i_start, l_word):
    assert i_start >= 0 and i_start < len(genome)
    if i_start + l_word < len(genome):
        return genome[i_start:i_start+l_word]
    else:
        l_remaining_1 = (i_start + l_word) % len(genome)
        n_loops = (l_word - l_remaining_1 - (len(genome)-i_start)) // len(genome)
        if n_loops == 0:
            genome_out = np.array([*genome[i_start:], *genome[0:l_remaining_1]])
            return genome_out
        else:
            l_remaining_2 = l_remaining_1 % len(genome)
            looping_genomes = [list(genome) for _ in range(n_loops)]
            looping_genomes = np.array(looping_genomes).flatten()
            genome_out = np.array([*genome[i_start:], *looping_genomes, *genome[0:l_remaining_2]])
            return genome_out

@njit
def compute_motif_entropies(genome, ls_motif):
    # construct complementary genome
    genome_comp = construct_complementary_genome(genome)
    
    # extract motif frequencies
    entropies = np.zeros(len(ls_motif))
    freqs_glob = []
    for il, l_motif in enumerate(ls_motif):
        vector__word_to_index = 4**np.arange(l_motif)[::-1]
        freqs_loc = np.zeros(4**l_motif)
        for i_start in range(len(genome)):
            word = extract_word(genome, i_start, l_motif)
            word_index = np.sum(vector__word_to_index * word)
            freqs_loc[word_index] += 1/(2*len(genome))
        for i_start in range(len(genome)):
            word = extract_word(genome_comp, i_start, l_motif)
            word_index = np.sum(vector__word_to_index * word)
            freqs_loc[word_index] += 1/(2*len(genome))
        freqs_glob.append(freqs_loc)
        indices_nonzero = np.argwhere(freqs_loc != 0)[:,0]
        entropy = -np.sum(freqs_loc[indices_nonzero]*np.log(freqs_loc[indices_nonzero]))
        entropies[il] = entropy/np.log(4)
    return entropies, freqs_glob

@njit
def compute_motif_entropies__wo_freqs(genome, ls_motif):
    # construct complementary genome
    genome_comp = construct_complementary_genome(genome)
    # extract motif frequencies
    entropies = np.zeros(len(ls_motif))
    for il, l_motif in enumerate(ls_motif):
        vector__word_to_index = 4**np.arange(l_motif)[::-1]
        freqs_loc = {}
        for i_start in range(len(genome)):
            word = extract_word(genome, i_start, l_motif)
            word_index = np.sum(vector__word_to_index * word)
            if not word_index in freqs_loc:
                freqs_loc[word_index] = 1/(2*len(genome))
            else:
                freqs_loc[word_index] += 1/(2*len(genome))
        for i_start in range(len(genome)):
            word = extract_word(genome_comp, i_start, l_motif)
            word_index = np.sum(vector__word_to_index * word)
            if not word_index in freqs_loc:
                freqs_loc[word_index] = 1/(2*len(genome))
            else:
                freqs_loc[word_index] += 1/(2*len(genome))
        freqs = np.array(list(freqs_loc.values()))
        entropy = -np.sum(freqs*np.log(freqs))
        entropies[il] = entropy/np.log(4)
    return entropies

@njit
def compute_motif_entropies__wo_freqs__OLD(genome, ls_motif):
    entropies, _ = compute_motif_entropies(genome, ls_motif)
    return entropies

@njit
def compute_number_of_missing_motifs(genome, ls_motif):
    _, freqs = compute_motif_entropies(genome, ls_motif)
    return np.array([np.sum(freqs[i]==0) for i in range(len(ls_motif))])

@njit
def identify_L1__single_genome(genome):
    L1 = 1
    _, freqs = compute_motif_entropies(genome, ls_motif=np.array([L1]))
    while np.all(freqs[0] != 0):
        L1 += 1
        _, freqs = compute_motif_entropies(genome, ls_motif=np.array([L1]))
    L1 -= 1
    return L1

@njit
def identify_L2__single_genome(genome):
    entropy_max = np.log(2*len(genome)) / np.log(4)
    L2 = 1
    entropy = compute_motif_entropies__wo_freqs(genome, ls_motif=np.array([L2]))
    while not np.isclose(entropy[0], entropy_max):
        L2 += 1
        entropy = compute_motif_entropies__wo_freqs(genome, ls_motif=np.array([L2]))
    return L2

# @njit
def extract_histogram_of_motif_frequencies_lmotif_small_L1max(genome, l_motif):
    l_gen = len(genome)
    assert l_motif <= np.log(l_gen)/np.log(4)
    _, [freqs] = compute_motif_entropies(genome, np.array([l_motif]))
    freqs_hist = {}
    for freq in freqs:
        if not freq in freqs_hist:
            freqs_hist[freq] =  1
        else:
            freqs_hist[freq] += 1
    freqs_hist = {key:freqs_hist[key] for key in sorted(list(freqs_hist.keys()))}
    return freqs_hist

def extract_human_interpretable_frequency_distribution_for_lmotif_smaller_L1max(genome, l_motif):
    l_gen = len(genome)
    assert l_motif <= np.log(l_gen)/np.log(4)
    _, [freqs] = compute_motif_entropies(genome, np.array([l_motif]))
    # array containing all the possible shifts 
    # shift is the number that count by how much the copy-number of a given motif 
    # is increase or decreased relative to a uniform motif distribution
    shifts = np.array([int(i) for i in np.arange(-(2*l_gen)/(4**l_motif), 2*l_gen*(1-1/(4**l_motif))+1, 1)])
    # array containing the human-interpretable frequencies
    freqs_hi = np.zeros(len(shifts))
    for j, shift in enumerate(shifts):
        freqs_hi[j] = np.sum(freqs==(1/4**l_motif)+shift/(2*l_gen))
    return freqs_hi, shifts

def compute_entropy_shift(shifts_type_and_number, l_gen, l_motif):
    # compute array containing the shifts in motif numbers
    motif_number_shifts = []
    for shift_type, number_shifts in shifts_type_and_number.items():
        motif_number_shifts.extend([shift_type for _ in range(int(number_shifts))])
    motif_number_shifts = np.asarray(motif_number_shifts)
    x = 1+(motif_number_shifts*4**l_motif)/(2*l_gen)
    entropy = np.sum(-1/(4**l_motif) * x[x>0]*np.log(x[x>0]) / np.log(4))
    return entropy

def compute_entropy_given_number_redundant_motifs(n_rdndnt_mtf, l_gen):
    return ((1-2*n_rdndnt_mtf/(2*l_gen))*np.log(2*l_gen) \
           + 2*n_rdndnt_mtf/32*np.log(l_gen))/np.log(4)

def compute_number_redundant_motifs_given_entropy(entropy, l_gen):
    return l_gen*(entropy*np.log(4) - np.log(2*l_gen))/(np.log(l_gen)-np.log(2*l_gen))

def write_genomes_and_entropies_to_file(genomes, entropies, l_gen, ls_motif, \
                                        dirpath='./', filename=''):
    ls_motif_all = np.arange(1, np.max(ls_motif)+1, 1)
    if dirpath[-1] != '/':
        dirpath += "/"
    if filename == '':
        filename = f'genomes_and_entropies__lgen_{l_gen}__lsmotif_{",".join([str(l) for l in ls_motif])}.txt'
    filepath = dirpath + filename
    genomes_keys = np.array([str(int(''.join([str(el) for el in genomes[i]]), base=4)) \
                             for i in range(len(genomes))])
    filestring = "GENOME\tENTROPIES for motif lengths [%s]\n" %(' '.join(map(str, ls_motif_all)))
    for i in range(len(genomes_keys)):
        filestring += (f"{genomes_keys[i]:s}\t" + \
                   " ".join([f"{entropies[i][j]:.15e}" for j in range(len(entropies[i]))]) + \
                   "\n")
    f = open(filepath, 'w')
    f.write(filestring)
    f.close()

def read_genomes_and_entropies_from_file(filepath, l_gen):
    data_prelim = np.loadtxt(filepath, skiprows=1)
    if len(data_prelim.shape) == 2:
        len_data = data_prelim.shape[-1]
        data = np.loadtxt(filepath, skiprows=1, \
            dtype=[('genome','U1000'), *[('entropy_%s' %(i+1),'f8') for i in range(len_data-1)]])
        genome_keys = data['genome']
        genomes = []
        for i, genome_key in enumerate(genome_keys):
            genomes.append(np.fromiter(np.base_repr(int(genome_key), base=4).zfill(l_gen), dtype=int))
        genomes = np.asarray(genomes)
        entropies = np.array([data['entropy_%s' %(i+1)] for i in range(len_data-1)]).T
        return genomes, genome_keys, entropies
    elif len(data_prelim.shape) == 1:
        len_data = data_prelim.shape[-1]
        data = np.loadtxt(filepath, skiprows=1, \
            dtype=[('genome','U1000'), *[('entropy_%s' %(i+1),'f8') for i in range(len_data-1)]])
        genome_key = data['genome']
        genome = np.fromiter(np.base_repr(int(genome_key), base=4).zfill(l_gen), dtype=int)
        entropies = np.array([data['entropy_%s' %(i+1)] for i in range(len_data-1)]).T
        return genome, genome_key, entropies
