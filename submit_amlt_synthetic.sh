#!/bin/bash

################################################################
# Submits all training jobs for a model for FindManager        #
# and overrides prompts, so it is dangerous! 		       #
################################################################

FXN=$1
SEED=$2
MODEL=$3
for num in 500 1000 2000 5000 10000 12000;
do
		#fn_num=$((${num}/2))
	for fn_num in 1 5 10 20 50;
	do 
		# -r replaces existing experiment
		# -y answers yes to all prompts 
		cmd_str="amlt run -r -y -d \"train ${MODEL} ${FXN} ${SEED} ${num}_${fn_num}\" amlt_configs/${MODEL}/${FXN}_${SEED}_seed/${num}_${fn_num}.yaml :train ${MODEL}_${FXN}_${num}_${fn_num}_seed_${SEED}"
		echo "${cmd_str}"
		eval ${cmd_str}
	done
done

