#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/intent/logs/train_source.out
#SBATCH -p brtx6
#SBATCH --gpus=1

#FXN=$1
#MODEL=$2
#SEED=$3
#TEMP=$5
#DEVICE=$4

checkpoint_root="/srv/local1/estengel/${MODEL}_${TEMP}/${FXN}/${SEED}_seed"

for fxn_num in 15 30 75 7
do
    for num in 750 1500 3000 7500 15000 18000 
    do
        checkpoint_dir="${checkpoint_root}/${num}_${fxn_num}"
        mkdir -p ${checkpoint_dir}
        python -u main.py \
            --split-type interest \
            --bert-name bert-base-cased \
            --checkpoint-dir ${checkpoint_dir} \
            --batch-size 256 \
            --split-type interest \
            --total-train ${num} \
            --total-interest ${fxn_num} \
            --epochs 200 \
            --intent-of-interest ${FXN} \
            --mask-source-triggers \
            --mask-temperature ${TEMP} \
            --masking-method intent \
            --seed ${SEED} \
            --device 0 | tee ${checkpoint_dir}/stdout.log 
    done
done


