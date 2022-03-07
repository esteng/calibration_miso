#!/bin/bash


# submit each one; this will take a long time
for fxn in FindManager PlaceHasFeature FenceAttendee Tomorrow DoNotConfirm 
do
    ./submit_amlt_all.sh dangerous ${fxn} transformer_albert
done 

