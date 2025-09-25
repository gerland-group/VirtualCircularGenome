# Genome sampling

The source code in this directory is used to construct the circular genomes by employing a Metropolis-Hastings type algorithm as described in the Methods section of the paper.

## Directories
 - ```analysis/```: This directory contains the scripts used to run the sampling and analysis of the genomes.
 - ```outputs/```: The outputs of the analysis scripts are written to this directory.
 - ```src/```: This directory contains the source code accessed by the analysis scripts.

## Execution
Navigate to the ```analysis/```-directory and run the script with ```python3 <scriptname>```.
 - ```construct_genomes_via_metropolis_hastings__maximally_uniform.py```: This script samples genomes of length ```l_gen``` that attain maximal motif entropies for given motif lengths (the motif lengths are set by the variable ```ls_motif```). The user can also input the characteristic length scales ($L_{\rm E}$ and $L_{\rm U}$, for definition see the paper), and the script checks that the sampled genomes obey these desired length scales. In the paper, this script was used to sample the unbiased genomes (genomes in which $L_{\rm U} = L_{\rm U}^{\rm min}$ and $L_{\rm E} = L_{\rm E}^{\rm max}$) for genome length 16 nt and 64 nt.
 - ```construct_genomes_via_metropolis_hastings__strongly_nonuniform.py```: This script reads genomes with maximally unbiased genomes (length scales $L_E = L_{\rm E}^{\rm max}$ and $L_U = L_{\rm U}^{\rm min}$) as input, and produces genomes with characteristic length scales $L_{\rm E} < L_{\rm E}^{\rm max}$ and $L_U > L_{\rm U}^{\rm min}$. To this end, the script performs Metropolis-Hasings sampling that reduces the motif entropy on the length scale $L_{\rm E} \leq L \leq L_{\rm U}$. The motif lengths for which the entropy is supposed to be minimized is set by ```ls_motif__mut```. For motifs with lengths ```ls_motif__immut```, the entropy is enforced to retain its (maximal) value. The Metropolis-Hastings algorithm is run for many Monte Carlo steps (here, ```N_steps = 3e6```) to allow the entropy on intermediate length scales to converge towards its minimum, ensuring a strong bias in the motif distribution.
 - ```construct_genomes_via_metropolis_hastings__weakly_nonuniform.py```: Like the previous script, this script reads maximally unbiased genomes and produces genomes with $L_{\rm E} < L_{\rm E}^{\rm max}$ and $L_U > L_{\rm U}^{\rm min}$. However, the algorithm terminates once a genome has reached the desired length scales $L_{\rm E}$ and $L_{\rm U}$ (not running for ```N_steps``` Monte-Carlo steps). That way, the reduction in motif entropy is small: The genomes have a weakly non-uniform motif frequency distribution.
 - ```read_genomes__*.py```: These scripts are used to read the output obtained by running the first three scripts. They compute the motif entropy of the genomes, and by sort the obtained genomes by their motif entropies.

## Outputs
All outputs are written to the ```outputs/```-directory. The directory contains data for genomes with length 16 nt (only maximally unbiased genomes) and 64 nt (maximally unbiased genomes, as well as genomes with weak and strong motif frequency bias).
 - Files with the name ```genomes_and_entropies__*``` contain a list of sampled genomes and corresponding motif entropies. To minimize memory consumption, we save the genome key instead of the genome sequence. The genome key is obtained by interpreting the genome as a number in base 4 and computing the corresponding integer in decimal representaiton. The ```read_genomes__*.py``` automatically convert the genome_key information to a human-readable genome sequence.
 - Files with the name ```entropies_timeevol__*``` are obtained from the script ```construct_genomes_via_metropolis_hastings__strongly_nonuniform.py``` and store information about the total motif entropy as a function of the steps in the Metropolis-Algorithm (used to ensure convergence).
