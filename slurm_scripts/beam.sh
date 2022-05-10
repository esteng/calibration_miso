#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/logs/beam.out 
#SBATCH -p brtx6
#SBATCH --gpus=1

# need CHECKPOINT_DIR, TRAINING_CONFIG, TEST_DATA, FXN set 

#./experiments/calflow.sh -a train 

./experiments/calflow.sh -a beam

