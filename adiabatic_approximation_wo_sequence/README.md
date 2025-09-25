# Adiabatic Approximation without Sequence Resolution

The source code in this directory is used to compute the replication yield, fidelity and efficiency of a VCG under the assumption that hybridization and dehybridization of complexes is far quicker than any ligation reactions (adiabatic approximation). To compute replication observables, the code computes the equilibrium concentrations of single strands and complexes in the solution using on the binding affinities determined from a simplified effective nearest neighbor model for hybridiation energies. In this program, we do not resolve different oligomer sequences individually, but only distinguish oligomers based on their length; all oligomers with the same length are assumed to have the same concentration, regardless of the sequence. We account for different sequence identies by including combinatorial prefactors in the mass conservation laws (for details, see Methods section of the paper).

## Directories
 - ```analysis/```: This directory contains the scripts used to produce the plots shown in the paper.
    * ```changing_LV/```: contains the scripts to analyze the effect of changing the length of VCG oligomers on replication performance (Figure 2).
    * ```changing_LVmin_LVmax/```: contains the scripts to analyze the replication performance of VCG pools with a broad range of VCG oligomer lengths (Figures 3 and 4).
    * ```changing_LF/```: contains the scripts to analyze the effect of the feedstock on replication performance (Figure 5).
    * ```kinetic_suppression/```: contains the scripts to analyze the effect of kinetically suppressed ligations (Figures 6 and 7).
 - ```inputs/```: This directory contains the data obtained from the full kinetic simulation. It is used as input to produce Figure 1.
 - ```outputs/```: The outputs of the analysis scripts (plots and data files) are written to this directory. The subdirectories follow the same structure as the in the ```analysis/```-directory. 
 - ```src/```: This directory contains the source code accessed by the analysis scripts.

## Execution
Navigate to respective subdirectory in the ```analysis/```-directory and run the script with ```python3 <scriptname>```.
 - in ```changing_LV/```:
   * ```dataproduction__scaling_with_genome_length.py```: computes the oligomer length required for high efficiency replication as a function of genome length (data shown in Figure 1H).
   * ```main__influence_of_LV.py```: runs all calculations necessary for Figure 1 and produces Figure 1. Only the data for panel 1H is computed in advance and loaded to save compute time (vide supra).
   * ```SI__scaling_effective_association_constants.py```: produces the scaling plot in Appendix 1 - figure 1.
   * ```src__*```: source-code used by all analysis scripts in this subdirectory.
 - in ```changing_LVmin_LVmax/```:
   * ```main__length_modulated_facilitated_ligation.py```: analyze replication for a pool containing monomers, and VCG oligomers of two lengths (e.g. 4-mer and 8-mer). The script produces Figure 3 and the corresponding figure supplements.
   * ```main__influence_LVmin_LVmax.py```: script used to produce Figure 4. For ```READ = False``` (line 424), the script runs the analysis and produces the plot based on the computed data. With ```READ = True```, one can read pre-computed data and reduce the runtime of the script significantly, but the files to be read are too large to be contained within this repository. If all ```WRITE``` flags are set to true, the script produces the corresponding output files, such that the results can be read in the next execution of the script.
 - in ```changing_LF/```:
   * ```main__influence_LF.py```: script used to study the effect of the feedstock molecules on replication efficiency, creates Figure 5.
   * ```SI__scaling_effective_association_constants.py```: produces the scaling plot in Appendix 3 - figure 1.
 - in ```kinetic_suppression/```:
   * ```main__no_inversion_of_productivity.py```: analyzes replication if templated ligations are kinetically suppressed compared to monomer extension. For VCG pools containing a single length of VCG oligomers, longer oligomers are more productive (no inversion of productivity). The script produces Figure 6.
   * ```main__inversion_of_productivity.py```: analyzes replication with in pools with kinetically suppressed templated ligation. Given a range of VCG oligomers in the pool, short oligos can be more productive than longer ones (productivity inversion). The script produces Figure 7.
   * ```SI__approximation_equilibrium_concentrations.py```: semi-analytical approximation for the hybridization equilibrium used to compute the threshold concentrations for productivity inversion. Produces Appendix 5 - figure 1.
   * ```SI__inversion_of_productivity_for_Szostak_parameters.py```: analyzes wheter inversion of productivity is also predicted by our model for the experimental setup studied by the Ding et al. in the Szostak lab, Appendix 6 - figure 1.
   * ```src__*```: source-code shared across all scripts in this subdirectory.
 
## Outputs
All outputs are written to the respective subdirectory in ```outputs/```. 
