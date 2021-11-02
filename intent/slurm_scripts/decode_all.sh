#!/bin/bash

for fxn in 15 16 27 50 66
do 
    for seed in 12 31 64 do
    do
        export FXN=${fxn}
        export SEED=${seed}
        export MODEL=intent_no_source
        sbatch slurm_scripts/decode.sh --export 
    done
done 
         
