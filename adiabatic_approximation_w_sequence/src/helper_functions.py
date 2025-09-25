#!/bin/env python3

import os,binascii
import numpy as np

def convert_AGCT_to_0123(str_in):

    str_out = ''
    for letter in str_in:
        if(letter=='A'):
            str_out += '0'
        elif(letter=='G'):
            str_out += '1'
        elif(letter=='C'):
            str_out += '2'
        elif(letter=='T'):
            str_out += '3'
        else:
            raise ValueError("invalid input in convert_0123_to_AGCT")

    return str_out


def convert_0123_to_AGCT(str_in):

    str_out = ''
    for letter in str_in:
        if(letter=='0'):
            str_out += 'A'
        elif(letter=='1'):
            str_out += 'G'
        elif(letter=='2'):
            str_out += 'C'
        elif(letter=='3'):
            str_out += 'T'
        else:
            raise ValueError("invalid input in convert_0123_to_AGCT")

    return str_out


def convert_AGCT_to_1234(str_in):

    str_out = ''
    for letter in str_in:
        if(letter=='A'):
            str_out += '1'
        elif(letter=='G'):
            str_out += '2'
        elif(letter=='C'):
            str_out += '3'
        elif(letter=='T'):
            str_out += '4'
        else:
            raise ValueError("invalid input in convert_1234_to_AGCT")

    return str_out


def convert_1234_to_AGCT(str_in):

    str_out = ''
    for letter in str_in:
        if(letter=='1'):
            str_out += 'A'
        elif(letter=='2'):
            str_out += 'G'
        elif(letter=='3'):
            str_out += 'C'
        elif(letter=='4'):
            str_out += 'T'
        else:
            raise ValueError("invalid input in convert_1234_to_AGCT")

    return str_out


def construct_complementary_strand_rev2fwd(seq_in_3t5):

    ct = {'A':'T','T':'A','C':'G','G':'C'}

    seq_out_5t3 = ''
    for letter in seq_in_3t5:
        seq_out_5t3 += ct[letter]
    
    return seq_out_5t3


def construct_complementary_strand_fwd2fwd(seq_in_5t3):

    seq_out_3t5 = construct_complementary_strand_rev2fwd(seq_in_5t3)
    seq_out_5t3 = seq_out_3t5[::-1]

    return seq_out_5t3


def check_complementarity_single_letter(l1, l2):
    ct = {'A':'T','T':'A','C':'G','G':'C'}
    if(l1 == ct[l2]):
        return True
    else:
        return False


def convert_spacelineAGCT_to_123456(str_in):

    ct = {' ':1,'-':2,'A':3,'G':4,'C':5,'T':6}

    str_out = ""
    for letter in str_in:
        str_out += str(ct[letter])
    return str_out


def convert_123456_to_spacelineAGCT(str_in):

    ct = {"1":' ',"2":'-',"3":'A',"4":'G',"5":'C',"6":'T'}

    str_out = ""
    for letter in str_in:
        str_out += str(ct[letter])
    return str_out



def add_or_increase_in_dict(key, dictionary, value=1):

    if(not key in dictionary):
        dictionary[key] = value
    
    else:
        dictionary[key] += value


def add_or_include_in_dict(key, dictionary, value):

    if(not key in dictionary):
        dictionary[key] = [value]

    else:
        dictionary[key].append(value)


def read_pointer(object):

    return object.__repr__().split('at ')[1].split('>')[0]


def replace_str(str_basis, str_insert, index):

    str_out = str_basis[0:index] + str_insert + str_basis[index+len(str_insert):]
    elements_removed = list(set(list(str_basis[index+1:index+len(str_insert)])))

    if( (len(str_out) == len(str_basis)) and (len(elements_removed) == 0) ):
       return str_out
    elif( (len(str_out) == len(str_basis)) and (len(elements_removed) == 1) \
         and (elements_removed[0] == ' ') ):
        return str_out
    else:
        print(len(str_out), len(str_basis))
        print(elements_removed)
        raise ValueError("invalid string produced in replace_str")


def create_random_hexstring():
    return str(binascii.b2a_hex(os.urandom(12)))[2:-1]


def build_scientific_notation(value, decimals):

    power = int(np.floor(np.log10(value)))
    prefactor = np.round(value / 10**power, decimals)
    formatter = "%" + "1.%df" %decimals
    prefactor_str = eval(formatter %prefactor)
    value_str = "%sE%d" %(prefactor_str, power)

    return value_str
