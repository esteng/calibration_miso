#!/bin/bash

# make the function test files 
python scripts/get_fxn_file.py ~/resources/data

# upload data, use r to replace current 
amlt upload --config-file amlt_configs/transformer/FindManager_12_seed/max_100.yaml

# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    for seed in 12 31 64
    do  
        ./decode_amlt_beam.sh ${fxn} ${seed} transformer 
    done
done 
