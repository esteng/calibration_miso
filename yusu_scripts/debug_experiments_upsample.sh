#!/bin/bash


MODEL=transformer_upsample_constant
FXN=FindManager
SEED=12
fn_num=100
num=10000

cmd_str="amlt run -r -y -d \"train ${MODEL} ${FXN} ${SEED} ${num}_${fn_num}\" amlt_configs/${MODEL}/${FXN}_${SEED}_seed/${num}_${fn_num}.yaml :train ${MODEL}_${FXN}_${num}_${fn_num}_seed_${SEED}"

echo "${cmd_str}"
eval ${cmd_str}

