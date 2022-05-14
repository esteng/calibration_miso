#!/bin/bash

# ARG1: FACTOR
for seed in 12 31 64
do
    #for fxn in 50 66 15 16 27
    for fxn in 15 
    do
        export FXN=$fxn
        export SEED=$seed
        export MODEL=intent_upsample_constant_no_source
        sbatch slurm_scripts/train_intent_upsample_constant_no_source_${fxn}.sh --export 
    done
done
