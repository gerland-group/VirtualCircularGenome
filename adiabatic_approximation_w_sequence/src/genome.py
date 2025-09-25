#!/bin/env python3

from numba import njit
import numpy as np

import helper_functions as hf

def generate_periodic_substring(string, istart, l):

    if(istart>=len(string)):
        raise ValueError('invalid istart in generate_periodic_substring')

    if(istart+l<=len(string)):
        return string[istart:istart+l]
    
    else:
        lpart1 = len(string)-istart
        npart2 = int((l-lpart1)/len(string))
        lpart2 = npart2 * len(string)
        lpart3 = l-lpart1-lpart2

        return string[istart:]+npart2*string+string[0:lpart3]

@njit
def compute_complementary_sequence_key2key(seq1_key, alphabet, length):
    seq2_key = alphabet**length - 1 - seq1_key
    return seq2_key

def compute_complementary_sequence_key2key__no_njit(seq1_key, alphabet, length):
    seq2_key = alphabet**length - 1 - seq1_key
    return seq2_key


class Genome:

    def __init__(self, seq1_key, alphabet, length_both_strands, lmax):

        # alphabet size
        self.alphabet = alphabet

        # lenght of genome
        self.length = length_both_strands
        self.length_ss = length_both_strands//2

        # genome number
        self.seq1_key = int(seq1_key)
        
        # sequences
        self.seq1_5t3_numb = np.base_repr(self.seq1_key, alphabet).zfill(self.length//2)
        self.seq2_key = compute_complementary_sequence_key2key__no_njit(self.seq1_key, self.alphabet, self.length//2)
        self.seq2_3t5_numb = np.base_repr(self.seq2_key, alphabet).zfill(self.length//2)
        self.seq2_5t3_numb = self.seq2_3t5_numb[::-1]
        
        # word lengths
        self.lmax = lmax
        self.ls = np.arange(1, self.lmax+1, 1)


    def list_included_words_single_length(self, l, KEY=False):

        words_sl = {}

        for i in range(len(self.seq1_5t3_numb)):
            if(KEY):
                it = tuple([int(k) for k in np.base_repr(i, base=self.alphabet).zfill(l)])
                if not it in words_sl:
                    words_sl[it] = 1
                else:
                    words_sl[it] += 1
            else:
                word = hf.convert_0123_to_AGCT(generate_periodic_substring(\
                    self.seq1_5t3_numb, i, l))
                if not word in words_sl:
                    words_sl[word] = 1
                else:
                    words_sl[word] += 1
        
        for i in range(len(self.seq2_5t3_numb)):
            if(KEY):
                it = tuple([int(k) for k in np.base_repr(i, base=self.alphabet).zfill(l)])
                if not it in words_sl:
                    words_sl[it] = 1
                else:
                    words_sl[it] += 1
            else:
                word = hf.convert_0123_to_AGCT(generate_periodic_substring(\
                    self.seq2_5t3_numb, i, l))
                if not word in words_sl:
                    words_sl[word] = 1
                else:
                    words_sl[word] += 1
        
        return words_sl
    

    def list_included_words(self, KEY=False):

        self.included_words = {}

        for l in self.ls:
            self.included_words[l] = self.list_included_words_single_length(l, KEY=KEY)


    def extend_list_included_words(self, l, KEY=False):
        self.included_words[l] = self.list_included_words_single_length(l, KEY=KEY)

