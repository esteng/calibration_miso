#!/bin/bash


for seed in 12 31 64
do
	for split in 500 1000 2000 5000 10000 12000
	do
		for fsplit in 1 5 10 20 50 
		do

			python scripts/error_analysis.py \
				--gold ~/resources/data/synthetic_two_piece/seq2seq/Func2/${split}_${fsplit}/test.tgt \
				--pred ~/amlt_models/synthetic_two_function_seq2seq/Func2_${seed}_seed/${split}_${fsplit}/translate_output/test.tgt  \
				--input ~/resources/data/synthetic_two_piece/seq2seq/Func2/${split}_${fsplit}/test.src_tok \
				--fxn-of-interest Func2 \
				--incorrect-output-with-fxn ~/scratch/error_analysis/Func2/${seed}_seed/${split}_${fsplit}/incorrect_with_fxn \
				--incorrect-output-without-fxn ~/scratch/error_analysis/Func2/${seed}_seed/${split}_${fsplit}/incorrect_without_fxn \
				--correct-output ~/scratch/error_analysis/Func2/${seed}_seed/${split}_${fsplit}/correct \
				--synthetic
		done
	done
done
