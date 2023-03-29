#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/logs/miso_backtranslate.out
#SBATCH -p brtx6
#SBATCH --gpus=1

    

#./calibration_scripts/backtranslation_from_pred.sh \
#    /brtx/604-nvme1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/bart-large_calflow_last_user_all_0.0001/checkpoint-10000 \
#    /brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/translate_output_calibrated/dev_all.tgt

./calibration_scripts/backtranslation_from_pred.sh \
    ${REPHRASER} \
    /brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/translate_output_calibrated/dev_all.tgt
