## recipe for creating a HIT 

1. decode MISO with nucleus sampling `./experiments/calflow.sh -a eval_calibrate`
2. prepare output for translation `prep_for_translate.py`
3. translate to get paraphrases `translate.sh` 
4. prep for forced decode `prep_for_miso_forced_decode.py` 
5. forced decode `forced_decode.sh` 
6. rank paraphrases by loss `rank_by_loss.py`
7. prepare csv `prep_csv.py` 