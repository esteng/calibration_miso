#!/bin/bash

for seed in 12 31 64 84 88
do
	mkdir -p "amlt_configs/synthetic_seq2seq_two_min_pair/Func2_${seed}_seed/"
	for num in 500 1000 2000 5000 10000 12000
	do
		for split in 1 5 10 20 50
		#for split in 250 500 1000 2500 5000 6000
		do
			src_file="amlt_configs/synthetic_simple_vanilla_lstm/Func47_12_seed/10000_10.yaml"
			tgt_file="amlt_configs/synthetic_seq2seq_two_min_pair/Func2_${seed}_seed/${num}_${split}.yaml"
			cp ${src_file} ${tgt_file}
			sed -i "s/Func47/Func2/g" ${tgt_file}
			sed -i "s/synthetic_simple_vanilla_lstm/synthetic_two_function_min_pair/g" ${tgt_file}
			sed -i "s/\/synthetic_simple\//\/synthetic_two_piece\/seq2seq\//g" ${tgt_file}
			sed -i "s/12_seed/${seed}_seed/g" ${tgt_file}
			sed -i "s/10000_10/${num}_${split}/g" ${tgt_file}
		done
	done
done

