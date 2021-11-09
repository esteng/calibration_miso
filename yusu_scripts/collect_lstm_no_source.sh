#!/bin/bash

results_dir=$1/lstm_no_source

mkdir -p ${results_dir} 

# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    python scripts/collect_results.py --model-dir ~/amlt_models/vanilla_lstm_no_source --data-dir ~/resources/data/smcalflow.agent.data --out-path ${results_dir}/${fxn}_vanilla_lstm_no_source.csv --fxn ${fxn} --fxn-splits 100 --test 
done 
