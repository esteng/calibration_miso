#!/bin/bash

for seed in 12 31 64
do
	for num in 5000 10000 20000 50000 100000 max 
	do
		mv miso/training_config/calflow_transformer/EventAttendance/${seed}_seed/${num}_290.jsonnet miso/training_config/calflow_transformer/EventAttendance/${seed}_seed/${num}_897.jsonnet
	done
done
