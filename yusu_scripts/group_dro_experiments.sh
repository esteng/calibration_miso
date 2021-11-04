#!/bin/bash

# submit each one; this will take a long time
for seed in 12 31 64 
do 
    for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
    do
        ./submit_amlt_dangerous.sh ${fxn} ${seed} transformer_group_dro
    done 
done
