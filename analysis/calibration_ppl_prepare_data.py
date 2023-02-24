import json
import argparse
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
    elif "calflow" in bclamp_path:
        dataset = "calflow"
    else:
        dataset = "cosql"
    bart_data = read_benchclamp_file(bclamp_path)
    # bart_min_probs, bart_mean_probs, bart_accs = get_probs_and_accs_benchclamp(bart_data) 
    if dataset == "spider":
        input_test_data = read_benchclamp_file("/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/Spider/test_all.jsonl")
    elif dataset == "calflow":
        input_test_data = read_benchclamp_file("/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/test_all.jsonl")
    else:
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

    save_prefix = dataset

    if "bart-large" in bclamp_path:
        model = "bart-large"
    elif "bart-base" in bclamp_path:
        model = "bart-base"
    elif "codet5-base" in bclamp_path:
        model = "codet5-base"
    elif "t5-large" in bclamp_path:
        model = "t5-large"
    elif "t5-base" in bclamp_path and "codet5" not in bclamp_path:
        model = "t5-base"
    elif "t5-small" in bclamp_path:
        model = "t5-small"
    else:
        raise ValueError(f"model not found {bclamp_path}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spider")
    args = parser.parse_args()
    ece_metric = ECEMetric(n_bins=20, binning_strategy="adaptive")

    if args.dataset == "spider":

        paths = ["/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_spider_past_none_db_val_all_0.0001_5000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T031316.jsonl",
                "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-small-lm-adapt_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230203T092044.jsonl",
                "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-base-lm-adapt_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230206T093954.jsonl",
                "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-large-lm-adapt_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T064137.jsonl",
                "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-base_spider_past_none_db_val_all_0.0001_5000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T060905.jsonl",
                "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_spider_past_none_db_val_all_0.0001_5000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T031316.jsonl",
                "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/codet5-base_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T214405.jsonl"]
    elif args.dataset == "calflow":
        paths = ["/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-small-lm-adapt_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230223T160146.jsonl", 
                    "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230223T135433.jsonl", 
                    "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-large-lm-adapt_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20221102T103315.jsonl", 
                    "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-base_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230222T155549.jsonl", 
                    "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20221101T105421.jsonl"]
    for path in paths:
        if path == "":
            continue
        print(path)
        write_file(path, ece_metric)