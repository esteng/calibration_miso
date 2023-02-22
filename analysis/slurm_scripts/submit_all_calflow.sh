#!/bin/bash

# for path in calflow_test_by_bart-base_bin calflow_test_by_bart_bin calflow_test_by_bart-large_bin calflow_test_by_t5-base_bin calflow_test_by_t5-large_bin calflow_test_by_t5-small_bin 
for path in calflow_test_by_bart-large_bin  calflow_test_by_t5-large_bin 
do 
    echo $path
    export TEST_DIR=${path}
    sbatch slurm_scripts/train_calflow_lstm.sh --export 
    ls $path
done 
