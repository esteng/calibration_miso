#!/bin/bash

#for model in "/brtx/601-nvme1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000/" "/brtx/604-nvme1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/bart-large_calflow_last_user_all_0.0001/checkpoint-10000/" "/brtx/603-nvme1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-large-lm-adapt_calflow_last_user_all_0.0001/" 
for model in "/brtx/603-nvme1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-large-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000" 
do
    for split in "dev_all" "test_all"
    do 
        ./calibration_scripts/backtranslation.sh ${model} ${split} 
    done 
done
