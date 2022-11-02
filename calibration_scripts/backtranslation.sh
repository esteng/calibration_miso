#!/bin/bash 
# REQUIRES
# $1: calflow_to_text_model 
# $2: split name 

eval "$(conda shell.bash hook)"
# instructions for running backtranslation for cycle consistency 

echo "Setting up model and data paths"
# need a calflow-to-text model and a text-to-calflow model 
calflow_to_text_model=$1
text_to_calflow_model="/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data" 

# need calflow files in format for calflow_to_text model
dev_split=$2
data_dir="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/"
dev_input="${data_dir}/${dev_split}.jsonl"
gold_output_dir="/brtx/601-nvme1/estengel/resources/data/benchclamp/predicted/CalFlowV2/gold/"
pred_output_dir="${calflow_to_text_model}/for_roundtrip/"
mkdir -p $gold_output_dir
mkdir -p $pred_output_dir

# need to run calflow_to_text model on dev_input
cd /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm
echo "activate bclamp env"
conda deactivate 
conda activate bclamp 
export CHECKPOINT_DIR=${calflow_to_text_model}
export VALIDATION_FILE=${dev_input}
echo "running calflow_to_text model"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
./slurm_scripts/lispress_to_text/calflow/eval_hf.sh 

# need to postprocess prediction files into format for MISO prediction 
cd /home/estengel/incremental-function-learning 

echo "preparing data for miso"
# postprocess with the predicted file 
pred_inp_file="${CHECKPOINT_DIR}/outputs/generated_predictions.txt"
gold_inp_file=${dev_input}
pred_out_file="${pred_output_dir}/${dev_split}"
gold_out_file="${gold_output_dir}/${dev_split}"

# do predicted 
echo "preparing predicted file"
python calibration_scripts/prep_for_roundtrip.py \
    --text_pred_file ${pred_inp_file} \
    --data_jsonl ${gold_inp_file} \
    --out_file ${pred_out_file} \
    --out_format lines \

# do predicted gold
echo "preparing predicted file"
python calibration_scripts/prep_for_roundtrip.py \
    --text_pred_file ${pred_inp_file} \
    --data_jsonl ${gold_inp_file} \
    --out_file ${gold_out_file} \
    --out_format lines \
    --use_gold 

# run MISO on predicted 
echo "activate miso env"
conda deactivate 
conda activate miso_new 
export CHECKPOINT_DIR=${text_to_calflow_model}
export TEST_DATA=${pred_out_file}
export FXN="None"
model_basename=$(basename ${calflow_to_text_model})
model_parent_basename=$(basename $(dirname ${calflow_to_text_model}))
pred_miso_out_file="calibration_logs/miso_roundtrip_predicted_from_${model_parent_basename}-${model_basename}_${dev_split}.txt"
echo "running miso on predicted"
./experiments/calflow.sh -a eval_fxn > ${pred_miso_out_file}
filename=$(basename ${pred_out_file})
mv ${CHECKPOINT_DIR}/translate_output/${filename}.tgt ${calflow_to_text_model}/for_roundtrip/predicted_${filename}.tgt

# run MISO on gold
export TEST_DATA=${gold_out_file}
export FXN="None"
gold_miso_out_file="calibration_logs/miso_roundtrip_gold_${dev_split}.txt" 
echo "Running miso on gold"
./experiments/calflow.sh -a eval_fxn > ${gold_miso_out_file}
mv ${CHECKPOINT_DIR}/translate_output/${filename}.tgt ${calflow_to_text_model}/for_roundtrip/gold_${filename}.tgt

