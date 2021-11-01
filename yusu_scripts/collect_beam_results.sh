#!/bin/bash


# needs a results dir
results_dir=$1/transformer
mkdir -p ${results_dir} 

# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    for seed in 12 31 64
    do  
        cp ~/amlt_models/transformer/${fxn}_${seed}_seed/translate_output/fxn_dev_valid.tgt ${results_dir}/${fxn}_${seed}_transformer_beam_dev_valid.tgt 
    done
done 
