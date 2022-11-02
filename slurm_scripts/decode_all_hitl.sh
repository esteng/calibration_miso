#!/bin/bash

for thresh in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do 
    export HITL_THRESHOLD=${thresh} 
    sbatch slurm_scripts/decode_hitl.sh --export 
done 
