#!/bin/bash

# needs 
# $1 MISO checkpoint with logits decoded 
# $2 Backtranslate checkpoint 
# $3 split 

MISO_CKPT=$1
BACK_CKPT=$2
SPLIT=$3
DATA_DIR="hit/data/for_hit_round_2" 
# low-confidence examples replaced here with stratified sample 
# and already run through MISO 

# Prep for backtranslation 
echo "Preparing for backtranslation..."
python hit/scripts/prep_for_translate.py \
    --miso_pred_file hit/data/for_hit_round_2/stratified_data_by_bin.tgt \
    --out_file hit/data/for_translate/stratified_data_by_bin.jsonl \
    --n_pred 3 

# Backtranslate MISO output 
echo "Backtranslating MISO output..."
./calibration_scripts/translate_miso_output.sh \
    ${BACK_CKPT} \
    $(pwd)/hit/data/for_translate/stratified_data_by_bin.jsonl \
    $(pwd)/hit/data/translated_by_bart_large/

# Aggregate data and prep for HIT 
echo "Aggregating data and preparing for HIT..."
mkdir -p hit/data/for_hit/from_${SPLIT}/
python hit/scripts/convert_miso_output.py \
    --miso_pred_file ${MISO_CKPT}/translate_output_calibrated/${SPLIT}_data_by_bin.tgt \
    --translated_tgt_file hit/data/translated_by_bart_large/generated_predictions.txt \
    --out_dir hit/data/for_hit/from_${SPLIT}/ \
    --filter_fences
