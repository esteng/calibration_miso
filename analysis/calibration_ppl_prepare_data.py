import json
from collections import defaultdict 
import pathlib 

from calibration_metric import ECEMetric

from calibration_utils import (read_nucleus_file, 
                                read_gold_file,get_probs_and_accs, 
                                read_benchclamp_file, 
                                get_probs_and_accs_benchclamp,
                                get_probs_and_accs_sql,
                                get_accs_sql)

def write_file(bclamp_path, ece_metric):

    if "spider" in bclamp_path:
        dataset = "spider"
    else:
        dataset = "cosql"
    bart_data = read_benchclamp_file(bclamp_path)
    # bart_min_probs, bart_mean_probs, bart_accs = get_probs_and_accs_benchclamp(bart_data) 
    if dataset == "spider":
        spider_gold_path = "/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/Spider/test_all.jsonl"
        input_test_data = read_benchclamp_file("/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/Spider/test_all.jsonl")
    else:
        spider_gold_path = "/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CoSQL/test_all.jsonl"
        input_test_data = read_benchclamp_file("/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CoSQL/test_all.jsonl")

    bart_min_probs, bart_mean_probs, bart_exact_accs = get_probs_and_accs_benchclamp(bart_data) # , spider_gold_path) 

    (min_values_em, 
    min_bins, 
    min_bin_number) = ece_metric.adaptive_bin(bart_min_probs, bart_exact_accs)

    data_by_bin = defaultdict(list)
    for i, datum in enumerate(bart_data):
        input_datum = input_test_data[i]
        bin_number = min_bin_number[i]
        bin_confidence = min_bins[bin_number]
        bin_acc = min_values_em[bin_number]

        data_by_bin[bin_number].append((bin_confidence, bin_acc, datum, input_datum))

    if dataset == "spider":
        save_prefix = "spider"
    else:
        save_prefix = "cosql"
    if "bart-large" in bclamp_path:
        model = "bart-large"
    elif "bart-base" in bclamp_path:
        model = "bart-base"
    elif "code-t5" in bclamp_path:
        model = "code-t5"
    elif "t5-large" in bclamp_path:
        model = "t5-large"
    elif "t5-base" in bclamp_path:
        model = "t5-base"
    elif "t5-small" in bclamp_path:
        model = "t5-small"

    for bin_num in data_by_bin.keys():
        bin_conf = data_by_bin[bin_num][0][0]
        print(bin_num, bin_conf)
        bin_conf_str = f"{bin_conf:.2f}"
        bin_str = f"{bin_num}_{bin_conf_str}"
        # write the inputs to a file for later analysis 
        out_dir = pathlib.Path(f"{save_prefix}_test_by_{model}_bin")
        out_dir.mkdir(exist_ok=True)
        with open(f"{save_prefix}_test_by_{model}_bin/{bin_str}.jsonl","w") as f1:
            for (_, _, datum, input_datum) in data_by_bin[bin_num]:
                f1.write(json.dumps(input_datum) + "\n")

if __name__ == "__main__":

    ece_metric = ECEMetric(n_bins=20, binning_strategy="adaptive")

    paths = ["/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_spider_past_none_db_val_all_0.0001_5000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T031316.jsonl",
            "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-small-lm-adapt_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230203T092044.jsonl",
            "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-base-lm-adapt_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230206T093954.jsonl",
            "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-large-lm-adapt_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T064137.jsonl",
            "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-base_spider_past_none_db_val_all_0.0001_5000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T060905.jsonl",
            "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_spider_past_none_db_val_all_0.0001_5000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T031316.jsonl",
            "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/codet5-base_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T214405.jsonl"]

    for path in paths:
        print(path)
        write_file(path, ece_metric)