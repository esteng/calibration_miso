#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/hf_models/logs/train_calflow_large.out
#SBATCH -p brtx6
#SBATCH --gpus=1

# need CHECKPOINT_DIR, DATA_DIR

./hf_models/scripts/train_calflow_t5_large.sh
