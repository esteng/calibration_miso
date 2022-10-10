#!/bin/bash

# needs 
# $1 MISO checkpoint with logits decoded 
# $2 Backtranslate checkpoint 
# $3 split 

MISO_CKPT=$1
BACK_CKPT=$2
SPLIT=$3
DATA_DIR="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/" 
# extract low-confidence examples from logits file 
echo "Extracting low-confidence examples..."
python hit/scripts/extract_mistakes.py \
    --logit_file ${MISO_CKPT}/translate_output/${SPLIT}_all_losses.json \
    --source_file ${DATA_DIR}/${SPLIT}_all.src_tok \
    --target_file ${DATA_DIR}/${SPLIT}_all.tgt \
    --output_file hit/data/for_miso/${SPLIT}_data_by_bin.json \
    --threshold 0.85 \
      

# Prepare low-confidence examples for MISO translation 
echo "Preparing for MISO translation with nucleus beam search..."
mkdir -p hit/data/for_miso/${SPLIT}
python hit/scripts/prep_for_miso.py \
    --bin_file hit/data/for_miso/${SPLIT}_data_by_bin.json \
    --out_dir hit/data/for_miso/${SPLIT}/

# Translate low-confidence examples with MISO
export TEST_DATA=hit/data/for_miso/${SPLIT}/dev_data_by_bin
export FXN=None
export CHECKPOINT_DIR=${MISO_CKPT}
echo "Running MISO on ${TEST_DATA}..."
./experiments/calflow.sh -a eval_calibrate

# Prep for backtranslation 
echo "Preparing for backtranslation..."
python hit/scripts/prep_for_translate.py \
    --miso_pred_file ${MISO_CKPT}/translate_output_calibrated/${SPLIT}_data_by_bin.tgt \
    --out_file hit/data/for_translate/${SPLIT}_data_by_bin.jsonl 

# Backtranslate MISO output 
echo "Backtranslating MISO output..."
./calibration_scripts/translate_miso_output.sh \
    ${BACK_CKPT} \
    $(pwd)/hit/data/for_translate/${SPLIT}_data_by_bin.jsonl \
    $(pwd)/hit/data/translated_by_bart_large/

# Aggregate data and prep for HIT 
echo "Aggregating data and preparing for HIT..."
mkdir -p hit/data/for_hit/from_${SPLIT}/
python hit/scripts/convert_miso_output.py \
    --miso_pred_file ${MISO_CKPT}/translate_output_calibrated/${SPLIT}_data_by_bin.tgt \
    --translated_tgt_file hit/data/translated_by_bart_large/generated_predictions.txt \
    --out_dir hit/data/for_hit/from_${SPLIT}/ \
    --filter_fences
