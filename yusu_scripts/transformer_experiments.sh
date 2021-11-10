#!/bin/bash


# submit each one; this will take a long time
for fxn in FindManager Tomorrow DoNotConfirm 
do
    ./submit_amlt_all.sh dangerous ${fxn} transformer
done 
