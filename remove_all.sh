#!/bin/bash 

for job in transformer_DoNotConfirm_100000_100_seed_31 transformer_DoNotConfirm_max_100_seed_31 
do
	echo "amlt remove -y ${job}"
	amlt remove -y ${job} 
done
