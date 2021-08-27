#!/bin/bash

################################################################
# Submits all training jobs for a model for FindManager        #
# and overrides prompts, so it is dangerous! 		       #
################################################################

FXN=$1
SEED=$2
MODEL=$3
for num in 5000 10000 20000 50000 100000 max;
do
	#for fn_num in 50 100 200 500
	for fn_num in 100
	do 
		# -r replaces existing experiment
		# -y answers yes to all prompts 
		cmd_str="amlt run -r -y -d \"train ${MODEL} ${FXN} ${SEED} ${num}_${fn_num}\" amlt_configs/${MODEL}/${FXN}_${SEED}_seed/${num}_${fn_num}.yaml :train ${MODEL}_${FXN}_${num}_${fn_num}_seed_${SEED}"
		echo "${cmd_str}"
		eval ${cmd_str}
	done
done
