#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/intent/logs/full.out
#SBATCH -p brtx6
#SBATCH --gpus=1

#FXN=$1
#MODEL=$2
#SEED=$3
#DEVICE=$4

checkpoint_root="/srv/local1/estengel/intent_fixed_test/full/${SEED}_seed"


num=18000

echo "Visible: ${CUDA_VISIBLE_DEVICES}"
checkpoint_dir="${checkpoint_root}/${num}_${fxn_num}"
mkdir -p ${checkpoint_dir}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u main.py \
    --split-type random \
    --bert-name bert-base-cased \
    --checkpoint-dir ${checkpoint_dir} \
    --batch-size 256 \
    --split-type interest \
    --total-train ${num} \
    --total-interest -1 \
    --epochs 200 \
    --seed ${SEED} \
    --device 0 | tee ${checkpoint_dir}/stdout.log 


