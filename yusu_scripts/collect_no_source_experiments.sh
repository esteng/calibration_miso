#!/bin/bash

# results dir will contain all the csvs 

results_dir=$1/no_source

mkdir -p ${results_dir} 


# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    python scripts/collect_results.py --model-dir ~/amlt_models/transformer_no_source --data-dir ~/resources/data/smcalflow.agent.data --out-path ${results_dir}/${fxn}_transformer_no_source.csv --fxn ${fxn} --fxn-splits 100 --test 
    for seed in 12 31 64
    do 
        cp ~/amlt_models/transformer_no_source/${fxn}_${seed}_seed/translate_output/test_valid.tgt ${results_dir}/${fxn}_${seed}_seed_test_valid.tgt 
    done 
done 
