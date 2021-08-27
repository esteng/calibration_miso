#!/bin/bash

type=$1
fxn=$2
model=$3

command_invocation="./decode_amlt_${type}.sh"

for seed in 12 31 64 
do
	${command_invocation} ${fxn} ${seed} ${model}
done
