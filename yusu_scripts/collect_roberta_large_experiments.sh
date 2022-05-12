#!/bin/bash

# results dir will contain all the csvs 


results_dir=$1/roberta_large

mkdir -p ${results_dir} 


# run each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    python scripts/collect_results.py --model-dir ~/amlt_models/transformer_roberta_large --data-dir ~/resources/data/smcalflow.agent.data --out-path ${results_dir}/${fxn}_roberta_large.csv --fxn ${fxn} --fxn-splits 100 --test 
    #for seed in 12 31 64 
    #do 
    #    for split in 5000 10000 20000 50000 100000 max 
    #    do 
    #        cp ~/amlt_models/transformer_upsample_${num}/${fxn}_${seed}_seed/${split}_100/translate_output/dev_valid.tgt ${results_dir}/${fxn}_${split}_100_${seed}_seed_dev_valid.tgt 
    #        cp ~/amlt_models/transformer_upsample_${num}/${fxn}_${seed}_seed/${split}_100/translate_output/test_valid.tgt ${results_dir}/${fxn}_${split}_100_${seed}_seed_test_valid.tgt 
    #    done 
    #done 
done 
