#!/bin/bash

onmt_text_data_dir="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data" 
python -m dataflow.leaderboard.predict \
    --datum_id_jsonl ${onmt_text_data_dir}/test.datum_id \
    --src_txt ${onmt_text_data_dir}/test.src_tok \
    --ref_txt ${onmt_text_data_dir}/test.tgt \
    --nbest_txt ${CHECKPOINT_DIR}/translate_output/test.tgt \
    --nbest 1

mv predictions.jsonl ${CHECKPOINT_DIR}/translate_output/test_pred.jsonl
