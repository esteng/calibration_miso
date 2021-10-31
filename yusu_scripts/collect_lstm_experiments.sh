#!/bin/bash

results_dir=$1/no_source

mkdir -p ${results_dir} 

# submit each one; this will take a long time
for fxn in FenceAttendee PlaceHasFeature 
do
    python scripts/collect_results.py --model-dir ~/amlt_models/vanilla_lstm --data-dir ~/resources/data/smcalflow.agent.data --out-path ${results_dir}/${fxn}_vanilla_lstm_test.csv --fxn ${fxn} --fxn-splits 100 --test 
done 
