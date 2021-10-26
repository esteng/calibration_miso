#!/bin/bash

base="amlt_configs/transformer_no_source/"
for seed in 12 31 64
do
    for fxn in Tomorrow PlaceHasFeature DoNotConfirm FenceAttendee 
    do
        cp -r ${base}/FindManager_${seed}_seed  ${base}/${fxn}_${seed}_seed
        sed -i "s/FindManager/${fxn}/g" ${base}/${fxn}_${seed}_seed/*
    done
done
