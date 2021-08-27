#!/bin/bash

################################################################
# Submits all dcoding jobs for a model for synthetic data      #
# and overrides prompts, so it is dangerous! 		       #
################################################################

FXN=$1
MODEL=$2
for SEED in 12 31 64 84 88
do
	for num in 500 1000 2000 5000 10000 12000;
	do
		for fn_num in 1 5 10 20 50;
		do 
			# -r replaces existing experiment
			# -y answers yes to all prompts 
			cmd_str="amlt run -r -y -d \"decode ${MODEL} ${FXN} ${SEED} ${num}_${fn_num}\" amlt_configs/${MODEL}/${FXN}_${SEED}_seed/${num}_${fn_num}.yaml :decode_test ${MODEL}_${FXN}_${num}_${fn_num}_seed_${SEED}"
			echo "${cmd_str}"
			eval ${cmd_str}
		done
	done
done
