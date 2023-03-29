#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/logs/decode_hitl.out
#SBATCH -p brtx6
#SBATCH --gpus=1

# need CHECKPOINT_DIR, TRAINING_CONFIG, TEST_DATA, FXN, HITL_THRESHOLD

#./experiments/calflow.sh -a train 

./experiments/calflow.sh -a eval_hitl

