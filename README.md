# Toward stable replication of genomic information in pools of RNA molecules

(c) Ludwig Burger, 2025

This software was used in the manuscript "Toward stable replication of genomic information in pools of RNA molecules" by Ludwig Burger and Ulrich Gerland published in <em>eLife</em> (https://doi.org/10.7554/eLife.104043).

## Description
In the Virtual Circular Genome scenario, a circular genome is mapped to a pool of oligomers that collectively the information of the circular genome. The oligomers in the pool can undergo templated ligation or be extended by monomers. That way, new longer oligomers are formed at the expense of short feedstock molecules (monomers or short oligoemrs), effectively replicating the genetic information of the VCG. The software provided in this repository was used
1. to pick the circular genomes that are encoded in the pool (directory ```genome_sampling/``` ),
2. to study the replication behavior of the VCG in the adiabatic limit (i.e., under the assumption that association/dissociation is far quicker than ligation reactions). 

For 2., we developed two approaches, 
<ol type='a'>
<li> sequence-resolved approach which distinguishes between oligomers with different sequence (directory <code>adiabatic_approximation_w_sequence/</code>), and </li>
<li> a coarse-grained approach that only distinguishes oligomers based on their length but not their sequence, and accounts for the combinatorics of base pairing by indroducing combinatorial multiplicities (directory <code>adiabatic_approximation_wo_sequence/</code>) </li>
</ol>
The latter variant is used to produce most of the results in the main text, with the first variant only used to study the behavior of longer genomes with biases motif distributions.
The source code for the full kinetic simulation is not provided in this repository, but is available upon reasonable request.

## Executions
For more details on how to execute the code, please consider the ```README.md``` files in the respective directories.

## Requirements
 - [Matplotlib](https://matplotlib.org/)
 - [Numba](https://numba.pydata.org/)
 - [NumPy](https://numpy.org/)
 - [pandas](https://pandas.pydata.org/) with [odfpy](https://pypi.org/project/odfpy/)
 - [SciPy](https://scipy.org/)

