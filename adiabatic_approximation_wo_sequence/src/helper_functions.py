#!/bin/env python3

import numpy as np

def reshape_array_1D_to_2D(x1D, y1D, z1D):

    if not (len(x1D) == len(y1D) and len(x1D) == len(z1D) and len(y1D) == len(z1D)):
        raise ValueError('invalid input arrays')

    x2D = np.asarray(sorted(list(set(x1D))))
    y2D = np.asarray(sorted(list(set(y1D))))
    z2D = np.zeros((len(y2D),len(x2D)))

    for i1D in range(len(x1D)):
        i2D = np.where(x2D==x1D[i1D])[0][0]
        j2D = np.where(y2D==y1D[i1D])[0][0]
        z2D[j2D,i2D] = z1D[i1D]

    return x2D, y2D, z2D

def add_or_append_to_dict(key, dictionary, value):

    if not key in dictionary:
        dictionary[key] = [value]
    
    elif key in dictionary:
        dictionary[key].append(value)
    
    else:
        raise ValueError('unexpected behaviour in add_or_append_to_dict')
    

def solve_quartic_equation(d3, d2, d1, d0):

    # find the root of quartic equation of the form x⁴ + d3 x³ + d2 x² + d1 x + d0 = 0

    m = np.array([[0,1,0,0], [0,0,1,0], [0,0,0,1], [-d0, -d1, -d2, -d3]])
    roots = np.linalg.eigvals(m)

    return roots