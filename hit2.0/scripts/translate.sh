#!/bin/bash

# REQUIRES
# $1: calflow_to_text_model 
# $2: file from hit/scripts/prep_for_translate.py 
# $3: output_dir 

calflow_to_text_model=/brtx/604-nvme1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/bart-large_calflow_last_user_all_0.0001/checkpoint-10000 \
# input_file=$(pwd)/hit2.0/data/for_translate/dev_all_top_1_from_miso.jsonl \
input_file=$(pwd)/hit2.0/data/for_translate/short.jsonl \
output_dir=$(pwd)/hit2.0/data/translated_by_bart_large/

eval "$(conda shell.bash hook)"
echo "Setting up model and data paths"

# need to run calflow_to_text model on input file 
cd /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm
echo "activate bclamp env"
conda deactivate 
conda activate bclamp 
export CHECKPOINT_DIR=${calflow_to_text_model}
export VALIDATION_FILE=${input_file}
echo "running calflow_to_text model"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
./slurm_scripts/lispress_to_text/calflow/eval_hf_multiple.sh 

pred_out_file="${CHECKPOINT_DIR}/outputs/generated_predictions.txt"
echo "copying ${pred_out_file} to ${output_dir}" 
cp ${pred_out_file} ${output_dir} 

cd /home/estengel/incremental-function-learning 
echo "activate miso env"
conda deactivate 
conda activate miso_new 


    