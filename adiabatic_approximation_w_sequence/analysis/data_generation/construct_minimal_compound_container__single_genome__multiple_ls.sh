#!/bin/bash

# Lgen=64
Lgen=$(cat params.yaml | grep 'l_gen' | sed -E 's/:*\ /:/' | cut -d ":" -f 2)
L1=$(cat params.yaml | grep "L1" | sed -E 's/:*\ /:/' | cut -d ":" -f 2)
L2=$(cat params.yaml | grep "L2" | sed -E 's/:*\ /:/' | cut -d ":" -f 2)
bias=$(cat params.yaml | grep "bias" | sed -E 's/:*\ /:/' | cut -d ":" -f 2)
# extract ls_oligo_V from the parameter file
line=$(grep '^ls_oligo_V:' params.yaml);
ls_oligo_V=$(echo "$line" | sed -E 's/.*\[(.*)\]/\1/' | tr ',' ' ');

path="../../initial_configurations/Lgen_"$Lgen"__L1_"$L1"__L2_"$L2"__"$bias"/"
echo $path

echo "constructing compond containers for Lgen=$Lgen, L1=$L1, L2=$L2 and bias $bias"

for l_oligo in $ls_oligo_V;
    do echo "l_oligo: " $l_oligo;
    screen -dmS "($Lgen,$L1,$L2,$bias,[1,$l_oligo])" python3 construct_minimal_compound_container.py $Lgen $L1 $L2 "$bias" "[1,$l_oligo]";
done;
