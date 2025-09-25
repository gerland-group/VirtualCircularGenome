#!/bin/bash

cVtot=$1

ls_oligo_V=$(cat params.yaml | grep "ls_oligo_V" | sed -E 's/:*\ /:/' | cut -d ":" -f 2 | sed -E 's/\[(.*)\]/\1/' | tr "," " ");
for l_oligo_V in $ls_oligo_V;
  do echo "loligoV: " $l_oligo_V " cVtot: " $cVtot;
  screen -dmS "loligoV_$l_oligo_V cVtot_$cVtot" python3 read_minimal_container_and_compute_equilibrium_concentration_log.py $l_oligo_V $cVtot;
done
