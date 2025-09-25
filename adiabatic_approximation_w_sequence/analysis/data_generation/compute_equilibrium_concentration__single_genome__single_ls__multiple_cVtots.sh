#!/bin/bash

l_oligo_V=$1

for i in 1e-9 2.2e-9 4.8e-9 1e-8 2.2e-8 4.8e-8 1e-7 2.2e-7 4.8e-7 1e-6 2.2e-6 4.8e-6 1e-5 2.2e-5 4.8e-5 1.0e-4 2.2e-4 4.8e-4 1.0e-3; 
  do echo "loligoV: " $l_oligo_V " cVtot: " $i; 
  screen -dmS "loligoV_$l_oligo_V cVtot_$i" python3 read_minimal_container_and_compute_equilibrium_concentration_log.py $l_oligo_V $i;
done
