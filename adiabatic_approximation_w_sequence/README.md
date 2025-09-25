# Adiabatic Approximation with Sequence Resolution

The source code in this directory is used to compute the replication yield, fidelity and efficiency of a VCG under the assumption that hybridization and dehybridization of complexes is far quicker than any ligation reactions (adiabatic approximation). To compute replication observables, the code computes the equilibrium concentrations of single strands and complexes in the solution using on the binding affinities determined from a simplified effective nearest neighbor model for hybridiation energies. Unlike in the code without sequence resolution, we do distinguish between oligomers with different sequences in this code. To compute the hybridization equilibra, we need to enumerate all possible complexes that can form out of the existing oligomers, and find the equilibrium by solving law of mass action and mass conservation numerically.

## Directories
 - ```analysis/```: This directory contains the scripts to compute the replication observables and produce the plots (Appendix 2 - figures 1 and 2).
 - ```inputs/```: This directory contains the input data, e.g. the sequences of the genomes to be studied (these genomes have been picked with the ```genome_sampling``` scripts beforehand). 
 - ```outputs/```: The outputs of the analysis scripts (plots and data files) are written to this directory. 
 - ```src/```: This directory contains the source code accessed by the analysis scripts.

## Execution
The following steps are necessary to reproduce the data in the paper (please note that you can directly reproduce the plots by just running step 6):

1. Navigate to ```analysis/data_generation/```.
2. Set the parameters of interest in the ```params.yaml``` file. ```lgen``` specifies the length of the analyzed genome, ```L1``` and ```L2``` the characteristic genome length scales (exhaustive coverage length and unique motif length, see Methods), ```bias``` to specify whether the genome is supposed to have no (```none```), weak (```weakly_nonuniform```) or strong motif bias (```strongly_nonuniform```). Note that a ```genome_key.txt``` file for the genome with the desired properties must be provided in the ```inputs``` directory. Choose the length of VCG oligomers in the pool with ```ls_oligo_V```. Pools will only contain VCG oligomes of a single length as well as monomers. Providing multiple lengths here will create multiple pools with one length of VCG oligos each. 
3. Construct the corresponding VCG pool by running ```python3 construct_strands_from_genome.py```. The script reads the parameters from the ```params.yaml``` file. This creates the file ```complexes__ls_1_<LV>.txt``` in the input folder, with the file listing all oligomers in the VCG pool. The files produced that way are already included in the repo.
4. Enumerate all the complexes that can be formed out of the oligomers by running ```bash construct_minimal_compound_container__single_genome__multiple_ls.sh```. This scripts automatically enumerates all complexes (comprising three strands) for the VCG pools specified in the ```params.yaml``` file. Note that the script opens multiple subprocesses on your computer, that run the enumeration in the background. Once finished, the script produces files with the name ```minimal_container__ls_1_<LV>.pkl``` in the inputs folder. These files are not included in the repo due to their large size.
5. Compute the equilibrium concentration and the concentrations of productive complexes by running ```bash compute_equilibrium_concentration__single_genome__single_ls__multiple_cVtots.sh```. This script will automatically tune through a range of VCG concentrations for a single VCG pool and compute replication observables (multiple parallel processes in the background). The script writes files with the name ```concentrations_productive__cstot_<cFtot>_<cVtot>.txt``` to the ```outputs/```-directory. Note that it can only be run if the corresponding ```minimal_container__*.pkl``` file exists.
6. Repeat steps 1 through 5 for all genomes provided in the inputs-directory. 
6. Plot the data using the script in ```analysis/plotting```. 

## Outputs
The computed replication observables and the plots are written to the respective subdirectories in ```outputs/```.