#!/bin/bash 

#SBATCH -o /home/estengel/scratch/intent_train.out 
#SBATCH -p brtx6
#SBATCH --gpus=1

#FXN=$1
#MODEL=$2
#SEED=$3
#DEVICE=$4

checkpoint_root="/srv/local1/estengel/${MODEL}/${FXN}/${SEED}_seed"



for num in 750 1500 3000 7500 15000 18000 
do
    #for fxn_num in 7 15 30 75
    for fxn_num in 10000
    do
        echo "Visible: ${CUDA_VISIBLE_DEVICES}"
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
            --device 0 | tee ${checkpoint_dir}/stdout.log 
    done
done


