#!/bin/bash


# submit each one; this will take a long time
for fxn in Tomorrow DoNotConfirm 
do
    ./submit_amlt_all.sh dangerous ${fxn} transformer
done 

for fxn in FindManager
do
    for seed in 31 64 
    do 
        ./submit_amlt_dangerous.sh ${fxn} ${seed} transformer
    done

done
