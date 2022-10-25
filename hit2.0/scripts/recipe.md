## recipe for creating a HIT 

1. decode MISO with nucleus sampling `./experiments/calflow.sh -a eval_calibrate`
2. prepare output for translation `python hit2.0/scripts/prep_for_translate.py --miso_pred_file hit2.0/data/from_miso/dev_all.tgt --src_file /srv/local1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/dev_all.src_tok --tgt_file /srv/local1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/dev_all.tgt --idx_file /srv/local1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/dev_all.idx --out_file hit2.0/data/for_translate/dev_all_top_1_from_miso.jsonl --max_samples 500 --max_prob 0.6 --n_bins 10`
3. translate to get paraphrases `hit2.0/scripts/translate.sh hit2.0/data/for_translate_500_0.6/dev_all_top_1_from_miso.jsonl` 
4. rename the output file 
5. prep for forced decode ` python hit2.0/scripts/prep_for_miso_forced_decode.py --miso_pred_file hit2.0/data/for_translate/sampled_nucleus_lines.tgt --translated_file hit2.0/data/translated_by_bart_large/pilot_100_0.6.txt --n_preds 10 --out_dir hit2.0/data/for_forced_decode_pilot/` 
6. forced decode `hit2.0/scripts/forced_decode.sh hit2.0/data/for_forced_decode_pilot/dev_for_forced_decode` 
7. move the output from checkpoint dir to dir of your chosing in `hit2.0/data` 
8. rank paraphrases by loss `python hit2.0/scripts/rank_by_loss.py --loss_file hit2.0/data/from_forced_decode_pilot/dev_for_forced_decode_losses.json --translated_file hit2.0/data/translated_by_bart_large/pilot_100_0.6.txt --gold_dir /srv/local1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/ --gold_split dev_all --force_decode_input_dir hit2.0/data/for_forced_decode_pilot/ --out_file hit2.0/data/for_csv/dev_pilot.jsonl`
9. prepare csv `prep_csv.py` 