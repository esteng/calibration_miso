#!/bin/bash 

#SBATCH -o /home/estengel/incremental-function-learning/intent/logs/train.out
#SBATCH -p brtx6
#SBATCH --gpus=1

#FXN=$1
#MODEL=$2
#SEED=$3
#DEVICE=$4

#checkpoint_root="/brtx/603-nvme1/estengel/${MODEL}/${FXN}/${SEED}_seed"

checkpoint_root=${CHECKPOINT_DIR}



for fxn_num in 15 30 75 7
do
    for num in 750 1500 3000 7500 15000 18000 
    do
        checkpoint_dir="${checkpoint_root}/${SEED}_seed/${num}_${fxn_num}"
        #mkdir -p ${checkpoint_dir}
        echo "STARTING"
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
            --seed ${SEED} \
            --do-test-only \
            --special-test data/${num}_${fxn_num}_${SEED}_seed.json \
            --output-individual-preds \
            --device 0 
    done
done


