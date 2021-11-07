#!/bin/bash

# results dir will contain all the csvs 

results_dir=$1/group_dro

mkdir -p ${results_dir} 


# run each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    python scripts/collect_results.py --model-dir ~/amlt_models/transformer_group_dro --data-dir ~/resources/data/smcalflow.agent.data --out-path ${results_dir}/${fxn}_transformer_group_dro.csv --fxn ${fxn} --fxn-splits 100 --test 
done 
