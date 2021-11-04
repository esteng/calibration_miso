#!/bin/bash

for seed in 12 31 64
do
    for fxn in 66
    #for fxn in 66 15 16 27
    #for fxn in 50
    do
        export FXN=$fxn
        export SEED=$seed
        export MODEL=intent_no_source_manual_more
        #sbatch slurm_scripts/train_intent_no_source.sh --export 
        sbatch slurm_scripts/train_intent_${fxn}_no_trig.sh --export 
    done
done
