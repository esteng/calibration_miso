#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/intent/logs/conj.out
#SBATCH -p brtx6
#SBATCH --gpus=1


checkpoint_dir="/srv/local1/estengel/intent_conj_diverse/${RATIO}/${SEED}" 
mkdir -p ${checkpoint_dir} 

python -u main_conj.py \
    --train-data-path data/conj_data_diverse/${RATIO} \
    --devtest-data-path data/conj_data_diverse \
    --bert-name bert-base-cased \
    --checkpoint-dir ${checkpoint_dir} \
    --batch-size 256 \
    --epochs 200 \
    --intent-of-interest 50 \
    --seed ${SEED} \
    --device 0 | tee ${checkpoint_dir}/stdout.log 
