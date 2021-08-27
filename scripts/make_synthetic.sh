#!/bin/bash
fxn=$1
in_path=$2
out_path=$3

for total_n in 500 1000 2000 5000 10000 12000
do
    for sample_n in 1 5 10 20 50
    do
        output_dir="${out_path}/${fxn}/${total_n}_${sample_n}"
        mkdir -p ${output_dir}
        python ./scripts/sample_functions.py \
            --train-path ${in_path}/train \
            --out-path ${output_dir}/train \
            --fxn ${fxn} \
            --exact-n ${sample_n} \
            --total-n ${total_n} \

        cp ${in_path}/dev.* ${output_dir}
        cp ${in_path}/test.* ${output_dir}
    done
done

for sample_n in 5 10 20 50
do
    cp ${in_path}/dev.* ${out_path}/${fxn}/max_${sample_n}
    cp ${in_path}/test.* ${out_path}/${fxn}/max_${sample_n}
done
