#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/analysis/logs/2train_calflow_lstm.out 
#SBATCH --partition=brtx6-10,brtx6
#SBATCH --gpus=1


python calibration_sql_lstm_lm.py \
    --train_file /brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/train_all.jsonl \
    --dev_file /brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/dev_all.jsonl \
    --test_dir ${TEST_DIR}


python calibration_sql_lstm_lm.py \
    --train_file /brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/train_all.jsonl \
    --dev_file /brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/dev_all.jsonl \
    --test_dir ${TEST_DIR} \
    --use_target
