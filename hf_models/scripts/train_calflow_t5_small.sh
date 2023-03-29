# ./run.sh .00001 1
export BS=30; 
export USE_TF=0

python -u hf_models/train/trainer.py \
	--model_name_or_path t5-small \
	--output_dir ${CHECKPOINT_DIR}/calflow_t5_seq2seq \
	--do_train \
	--do_eval \
	--do_predict \
	--save_total_limit=2 \
	--train_file ${DATA_DIR}/calflow/train.jsonl \
	--validation_file ${DATA_DIR}/calflow/dev_valid.jsonl \
	--test_file ${DATA_DIR}/calflow/test_valid.jsonl \
	--predict_with_generate 1 \
	--learning_rate 1e-4 \
	--adam_eps 1e-06 \
	--overwrite_output_dir \
	--max_source_length 1024 \
	--max_target_length 512 \
	--per_device_train_batch_size $BS \
	--per_device_eval_batch_size $BS \
	--metric_for_best_model eval_exact_match \
	--greater_is_better=True \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 100 \
	--load_best_model_at_end=True \
	--save_strategy=steps \
	--logging_steps 1000 \
	--save_steps 1000 \
	--evaluation_strategy=steps \
	--eval_steps 1000 \
	--seed 42 

# deepspeed hf_models/train/trainer.py \
# 	--model_name_or_path t5-small \
# 	--output_dir ${CHECKPOINT_DIR}/calflow_t5_seq2seq \
# 	--do_train \
# 	--do_eval \
# 	--do_predict \
# 	--max_train_samples 100 \
# 	--max_eval_samples 100 \
# 	--save_total_limit=2 \
# 	--train_file ${DATA_DIR}/calflow/train.jsonl \
# 	--validation_file ${DATA_DIR}/calflow/dev_valid.jsonl \
# 	--test_file ${DATA_DIR}/calflow/test_valid.jsonl \
# 	--predict_with_generate 1 \
# 	--learning_rate 1e-4 \
# 	--adam_eps 1e-06 \
# 	--overwrite_output_dir \
# 	--max_source_length 1024 \
# 	--max_target_length 512 \
# 	--per_device_train_batch_size $BS \
# 	--per_device_eval_batch_size $BS \
# 	--metric_for_best_model eval_exact_match \
# 	--greater_is_better=True \
# 	--gradient_accumulation_steps 1 \
# 	--num_train_epochs 100 \
# 	--load_best_model_at_end=True \
# 	--save_strategy=steps \
# 	--logging_steps 1000 \
# 	--save_steps 1000 \
# 	--evaluation_strategy=steps \
# 	--eval_steps 1000 \
# 	--seed 42 \
# 	--deepspeed hf_models/scripts/deepspeed_configs/t5.json 

