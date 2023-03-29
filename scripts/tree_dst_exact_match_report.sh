# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

#########################################
#### Script to get official eval res ###
########################################

CHECKPOINT_DIR=$1
DATA_DIR=$2

nbest=1
dataflow_dialogues_stats_dir="${DATA_DIR}/stats"
dataflow_dialogues_dir="${DATA_DIR}"
onmt_translate_outdir="${CHECKPOINT_DIR}/translate_output"
evaluation_outdir="${CHECKPOINT_DIR}/evaluation_output"
mkdir -p "${evaluation_outdir}"

# compute stats 
mkdir -p "${dataflow_dialogues_stats_dir}"
python -m dataflow.analysis.compute_data_statistics \
        --dataflow_dialogues_dir ${dataflow_dialogues_dir} \
        --subset train valid \
        --outdir ${dataflow_dialogues_stats_dir}

# create the prediction report
python -m dataflow.onmt_helpers.create_onmt_prediction_report \
        --dialogues_jsonl ${dataflow_dialogues_dir}/valid.dataflow_dialogues.jsonl \
        --datum_id_jsonl ${dataflow_dialogues_dir}/valid.datum_id \
        --src_txt ${dataflow_dialogues_dir}/valid.src_tok \
        --ref_txt ${dataflow_dialogues_dir}/valid.tgt \
        --nbest_txt ${onmt_translate_outdir}/valid.tgt \
        --nbest ${nbest} \
        --outbase ${evaluation_outdir}/valid

# evaluate the predictions (all turns)
python -m dataflow.onmt_helpers.evaluate_onmt_predictions \
        --prediction_report_tsv ${evaluation_outdir}/valid.prediction_report.tsv \
        --use_leaderboard_metric \
        --scores_json ${evaluation_outdir}/valid.all.scores.json

# evaluate the predictions (refer turns)
python -m dataflow.onmt_helpers.evaluate_onmt_predictions \
        --prediction_report_tsv ${evaluation_outdir}/valid.prediction_report.tsv \
        --datum_ids_json ${dataflow_dialogues_stats_dir}/valid.refer_turn_ids.jsonl \
        --use_leaderboard_metric \
        --scores_json ${evaluation_outdir}/valid.refer_turns.scores.json

# evaluate the predictions (revise turns)
python -m dataflow.onmt_helpers.evaluate_onmt_predictions \
        --prediction_report_tsv ${evaluation_outdir}/valid.prediction_report.tsv \
        --datum_ids_json ${dataflow_dialogues_stats_dir}/valid.revise_turn_ids.jsonl \
        --use_leaderboard_metric \
        --scores_json ${evaluation_outdir}/valid.revise_turns.scores.json

