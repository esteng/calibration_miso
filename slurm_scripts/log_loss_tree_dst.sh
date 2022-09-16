#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/logs/log.out
#SBATCH -p brtx6
#SBATCH --gpus=1

# need CHECKPOINT_DIR, TRAINING_CONFIG, TEST_DATA, FXN set 


./experiments/tree_dst.sh -a log

