#!/bin/bash

for seed in 12 31 64 84 88
do
	for num in 500 1000 2000 5000 10000 12000 
	do
		for split in 1 5 10 20 50
		#for split in 250 500 1000 2500 5000 6000
		do
			#python scripts/make_configs.py --base-jsonnet-config miso/training_config/synthetic_simple_vanilla_lstm/Func47/12_seed/10000_10.jsonnet --model-type vanilla_lstm --function-type Func47 --data-split ${num}_${split} --json-out-path miso/training_config/synthetic_simple_vanilla_lstm/ --seed ${seed} 
			#python scripts/make_configs.py --base-jsonnet-config miso/training_config/calflow_mlp/overfit.jsonnet --model-type calflow_mlp --function-type Func2 --data-split ${num}_${split} --json-out-path miso/training_config/synthetic_two_function_single_output/ --seed ${seed} 

			#split=$(($num / 2))
			#python scripts/make_configs.py --base-jsonnet-config miso/training_config/synthetic_two_function_min_pair/base.jsonnet --model-type vanilla_calflow_parser --function-type Func2 --data-split ${num}_${split} --json-out-path miso/training_config/synthetic_two_function_min_pair/ --seed ${seed} 
			python scripts/make_configs.py --base-jsonnet-config miso/training_config/calflow_vanilla_transformer/base.jsonnet --model-type vanilla_transformer_calflow_parser --function-type Func2 --data-split ${num}_${split} --json-out-path miso/training_config/synthetic_transformer_two_function_seq2seq/ --seed ${seed} 
		done
	done
done
