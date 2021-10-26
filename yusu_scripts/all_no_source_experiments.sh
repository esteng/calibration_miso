#!/bin/bash

# make all the data
./scripts/make_all_subsamples_no_source.sh 

# upload data; hit the r key here for the replace option 
amlt upload --config-file amlt_configs/transformer_no_source/DoNotConfirm_12_seed/5000_100.yaml

# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    ./submit_amlt_all.sh dangerous ${fxn} transformer_no_source
done 
