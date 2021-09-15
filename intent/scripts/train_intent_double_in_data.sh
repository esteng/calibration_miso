#!/bin/bash 

FXN=$1
MODEL=$2
SEED=$3
DEVICE=$4

checkpoint_root="/srv/local1/estengel/${MODEL}/${FXN}/${SEED}_seed"

for num in 750 1500 3000 7500 15000 18000 
do
    for fxn_num in 7 15 30 75
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
            --epochs 100 \
            --intent-of-interest ${FXN} \
            --seed ${SEED} \
            --double-in-data \
            --device ${DEVICE} | tee ${checkpoint_dir}/stdout.log 
    done
done


