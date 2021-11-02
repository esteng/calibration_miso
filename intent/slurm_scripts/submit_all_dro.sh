#!/bin/bash

for seed in 12 31 64
do
    for fxn in 50 66 15 16 27
    do
        export FXN=$fxn
        export SEED=$seed
        export MODEL=intent_group_dro
        sbatch slurm_scripts/train_intent_group_dro.sh --export 
    done
done
