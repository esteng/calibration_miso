#!/bin/bash

# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    ./submit_amlt_all.sh dangerous ${fxn} transformer_upsample_32
    ./submit_amlt_all.sh dangerous ${fxn} transformer_upsample_16
    ./submit_amlt_all.sh dangerous ${fxn} transformer_upsample_64
done 
