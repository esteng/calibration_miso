#!/bin/bash 
# REQUIRES
# $1: calflow_to_text_model 
# $2: miso pred_file 

eval "$(conda shell.bash hook)"
# instructions for running backtranslation for cycle consistency 

echo "activate miso env"
conda deactivate 
conda activate miso_new 

echo "Setting up model and data paths"
# need a calflow-to-text model and a text-to-calflow model 
calflow_to_text_model=$1
text_to_calflow_model="/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data" 

# need calflow files in format for calflow_to_text model
initial_miso_parsed_file=$2
split=$(basename ${initial_miso_parsed_file} | cut -d'.' -f1)
data_dir="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/"
input_src_file=${data_dir}/${split}.src_tok
input_tgt_file=${data_dir}/${split}.tgt 

bclamp_data_dir="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/"
input_json_file=${bclamp_data_dir}/${split}.jsonl

translate_inp_file=${text_to_calflow_model}/translate_output_calibrated/${split}_for_backtranslate.jsonl

# convert calflow output to input for translation model 
python hit/scripts/prep_for_translate.py \
    --miso_pred_file ${initial_miso_parsed_file} \
    --src_file ${input_src_file} \
    --tgt_file ${input_tgt_file} \
    --n_pred 3 \
    --out_file ${translate_inp_file} 

output_dir="/brtx/601-nvme1/estengel/resources/data/benchclamp/predicted/CalFlowV2/level0/"
pred_output_dir="${calflow_to_text_model}/for_roundtrip_from_miso/"
mkdir -p $output_dir
mkdir -p $pred_output_dir

# need to run calflow_to_text model on dev_input
cd /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm
echo "activate bclamp env"
conda deactivate 
conda activate bclamp 
export CHECKPOINT_DIR=${calflow_to_text_model}
export VALIDATION_FILE=${translate_inp_file}
echo "running calflow_to_text model"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
./slurm_scripts/lispress_to_text/calflow/eval_hf.sh 

# need to postprocess prediction files into format for MISO prediction 
cd /home/estengel/incremental-function-learning

echo "preparing data for miso"
# postprocess with the predicted file 
pred_gloss_file="${CHECKPOINT_DIR}/outputs/generated_predictions.txt"
pred_lispress_file=${translate_inp_file}
pred_out_file="${pred_output_dir}/${split}"
gold_out_file="${output_dir}/${split}"

# do predicted 
echo "activate miso env"
conda deactivate 
conda activate miso_new 
echo "preparing predicted file"
echo ${pred_gloss_file}
echo ${input_json_file}
echo ${pred_out_file}
python calibration_scripts/prep_for_roundtrip.py \
    --text_pred_file ${pred_gloss_file} \
    --data_jsonl ${input_json_file} \
    --out_file ${pred_out_file} \
    --out_format lines \


# run MISO on predicted 
export CHECKPOINT_DIR=${text_to_calflow_model}
export TEST_DATA=${pred_out_file}
export FXN="None"
model_basename=$(basename ${calflow_to_text_model})
model_parent_basename=$(basename $(dirname ${calflow_to_text_model}))
pred_miso_out_file="calibration_logs/miso_roundtrip_predicted_from_${model_parent_basename}-${model_basename}_${split}.txt"
echo "running miso on predicted"
./experiments/calflow.sh -a eval_fxn > ${pred_miso_out_file}
filename=$(basename ${pred_out_file})
mv ${CHECKPOINT_DIR}/translate_output/${filename}.tgt ${calflow_to_text_model}/for_roundtrip_from_miso/predicted_${filename}.tgt

