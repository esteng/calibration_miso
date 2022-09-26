#!/bin/bash

pred_file=$1
gold_file=$2
out_file=$3

#pred_file="/srv/local1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000/outputs/generated_predictions.txt"
#gold_file="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/dev_medium.jsonl" 
#out_file="/brtx/601-nvme1/estengel/resources/data/benchclamp/predicted/CalFlowV2/gold/dev_medium" 

out_dir=$(dirname $out_file)
mkdir -p $out_dir

python scripts/prep_for_roundtrip.py \
    --text_pred_file ${pred_file} \
    --data_jsonl ${gold_file} \
    --out_file ${out_file} \
    --out_format lines \
    --use_gold