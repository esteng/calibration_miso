#!/bin/bash

# submit each one; this will take a long time
SEED=12
for seed in 12 31 64 
do 
    #for num in 16 32 64
    num=32
    for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
    do
        ./submit_amlt_dangerous.sh ${FXN} ${SEED} transformer_upsample_${num} 
    done 
done
