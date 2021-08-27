#!/bin/bash

FXN=$1

for num in 5000 10000 20000 50000 100000 max
do
	python minimal_pair_utils/extract_fxn_lines.py --train-path ~/resources/data/smcalflow_samples_big/${FXN}/${num}_100/ --write-path ~/resources/data/smcalflow_samples_big/${FXN}/${num}_100/ --fxn-of-interest ${FXN} --use-test
done
