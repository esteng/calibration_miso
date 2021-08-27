#!/bin/bash

FXN=$1
for seed in 12 31 64 
do
	mkdir -p amlt_configs/transformer/${FXN}_${seed}_seed/
	for num in 5000 10000 20000 50000 100000 max
	do
		for split in 50 200 500 
		#for split in 100
		do
			src_file="amlt_configs/transformer_upsampled/FindManager_12_seed/5000_200.yaml"
			tgt_file="amlt_configs/transformer/${FXN}_${seed}_seed/${num}_${split}.yaml"
			cp ${src_file} ${tgt_file}
			sed -i "s/transformer_upsampled/transformer/g" ${tgt_file}
			sed -i "s/FindManager/${FXN}/g" ${tgt_file}
			sed -i "s/12_seed/${seed}_seed/g" ${tgt_file}
			sed -i "s/5000_200/${num}_${split}/g" ${tgt_file}
			sed -i "s/smcalflow_samples_upsampled/smcalflow_samples_big/g" ${tgt_file}
			echo "wrote to ${tgt_file}"
		done
	done
done

