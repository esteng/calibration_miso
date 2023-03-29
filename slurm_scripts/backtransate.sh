#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/logs/backtranslate.out
#SBATCH -p brtx6
#SBATCH --gpus=1

BACK_CKPT="/brtx/604-nvme1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/bart-large_calflow_last_user_all_0.0001/checkpoint-10000"
SPLIT="dev"

./calibration_scripts/translate_miso_output.sh \
    hit/data/for_translate/${SPLIT}_data_by_bin.jsonl \
    hit/data/translated_by_bart_large/
    


