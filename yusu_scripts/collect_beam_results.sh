#!/bin/bash


# needs a results dir
results_dir=$1/transformer
mkdir -p ${results_dir} 

# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    for seed in 12 31 64
    do  
        for split in 5000 10000 20000 50000 100000 max 
        do 
            cp ~/amlt_models/transformer/${fxn}_${seed}_seed/${split}_100/translate_output/fxn_dev_valid_top_100.tgt ${results_dir}/${fxn}_${split}_100_${seed}_seed_transformer_dev_valid_top_100.tgt 
        done 
    done
done 
