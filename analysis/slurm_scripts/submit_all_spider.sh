#!/bin/bash

for path in spider_test_by_bart-base_bin spider_test_by_bart_bin spider_test_by_bart-large_bin spider_test_by_t5-base_bin spider_test_by_t5-large_bin spider_test_by_t5-small_bin spider_test_by_codet5-base_bin
do 
    echo $path
    export TEST_DIR=${path}
    sbatch slurm_scripts/train_spider_lstm.sh --export 
done 
