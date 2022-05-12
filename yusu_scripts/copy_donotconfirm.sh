#!/bin/bash 

for fxn in DoNotConfirm 
do
    for num in 5000 10000 20000 50000 100000 max 
    do 
        cp ~/resources/data/smcalflow.agent.data/test_valid.* ~/resources/data/smcalflow_samples_curated/${fxn}/${num}_100/
    done
done 
    
