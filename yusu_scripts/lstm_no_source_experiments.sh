#!/bin/bash


# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm FenceAttendee PlaceHasFeature 
do
    ./submit_amlt_all.sh dangerous ${fxn} vanilla_lstm_no_source
done 
