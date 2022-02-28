#!/bin/bash

# ARG1: FACTOR
for seed in 12 31 64
do
    for fxn in 50 66 15 16 27
    do
        export FXN=$fxn
        export SEED=$seed
        export MODEL=intent_no_shuffle
        export FACTOR=$1
        sbatch slurm_scripts/train_intent_all.sh --export 
    done
done
