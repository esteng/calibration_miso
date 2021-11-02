#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/intent/logs/source_intent_train.out 
#SBATCH -p brtx6
#SBATCH --gpus=1

#FXN=$1
#MODEL=$2
#SEED=$3
#DEVICE=$4

checkpoint_root="/srv/local1/estengel/intent_fixed_test/${MODEL}/${FXN}/${SEED}_seed"
FXN=50

for fxn_num in 75 30 15
do
    for num in 750 1500 3000 7500 15000 18000 
    do
        checkpoint_dir="${checkpoint_root}/${num}_${fxn_num}"
        mkdir -p ${checkpoint_dir}
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u main.py \
            --split-type interest \
            --bert-name bert-base-cased \
            --checkpoint-dir ${checkpoint_dir} \
            --batch-size 256 \
            --split-type interest \
            --total-train ${num} \
            --total-interest ${fxn_num} \
            --epochs 200 \
            --intent-of-interest ${FXN} \
            --seed ${SEED} \
            --do-source-triggers \
            --source-triggers radio,fm,am \
            --device 0 | tee ${checkpoint_dir}/stdout.log 
    done
done


