#!/bin/bash

# submit each one; this will take a long time
SEED=12
for num in 16 32 64
do
    for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
    do
        #./submit_amlt_all.sh dangerous_one_seed ${fxn} transformer_upsample_${num}
        ./submit_amlt_dangerous.sh ${FXN} ${SEED} transformer_upsample_${num} 
    done 
done
