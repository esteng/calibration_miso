#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/logs/train_FM1.out
#SBATCH -p brtx6
#SBATCH --gpus=1

# need CHECKPOINT_DIR, TRAINING_CONFIG, TEST_DATA, FXN set 

./experiments/calflow.sh -a resume 
./experiments/calflow.sh -a eval_fxn

