#!/bin/bash

for seed in 12 31 64
do
    for fxn in 50 66 15 16 27
    do
        export FXN=$fxn
        export SEED=$seed
        export MODEL=intent_no_source_fixed
        sbatch slurm_scripts/train_intent_${fxn}_no_trig_fixed.sh --export 
    done
done
