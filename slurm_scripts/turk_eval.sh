#!/bin/bash

#SBATCH -o /home/estengel/incremental-function-learning/logs/turk_eval.out
#SBATCH -p brtx6
#SBATCH --gpus=1

python hit/results/analyze_csv.py --csv hit/results/round1/Batch_4899871_batch_results.csv --json hit/data/for_hit/from_stratified/for_turk/data.json --do_iaa --do_rewrites
