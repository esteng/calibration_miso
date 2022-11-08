import json
import pathlib
from collections import defaultdict
import numpy as np 
import re 
from dataflow.core.lispress import parse_lispress, render_compact, render_pretty
from dataflow.core.linearize import lispress_to_seq

from calibration_utils import read_nucleus_file, read_gold_file, single_exact_match

def get_data(miso_path, bart_path, t5_path, gold_src_path, gold_tgt_path, gold_idx_path, gold_datum_id_path): 
    gold_src = read_gold_file(gold_src_path)
    gold_tgt = read_gold_file(gold_tgt_path)
    gold_idx = read_gold_file(gold_idx_path)
    with open(gold_datum_id_path) as f1:
        gold_datum_id = [json.loads(x) for x in f1.readlines()]
    assert(len(gold_tgt) == len(gold_idx))
    gold_tgt_by_idx = {idx: gold for idx, gold in zip(gold_idx, gold_tgt)}
    data_by_model = {}
    for model, path in zip(["miso", "bart", "t5"], [miso_path, bart_path, t5_path]):
        if model == "miso":
            __, miso_data = read_nucleus_file(path)
            assert(len(miso_data) == len(gold_tgt))
            data_by_model[model] = miso_data
        else:
            with open(path) as f1:
                data_by_model[model] = [json.loads(x) for x in f1.readlines()]
            assert(len(data_by_model[model]) == len(gold_tgt))
    return gold_tgt_by_idx, gold_src, gold_tgt, gold_idx, gold_datum_id, data_by_model

def get_low_prob(iterator, is_miso = False, threshold = 0.6):
    low_prob_idxs = []
    for idx, example in iterator:
        if is_miso: 
            try:
                min_prob = example[0][1]
                assert min_prob is not None
            except:
                min_prob = np.min(example[0]['expression_probs'])
                if min_prob is None:
                    min_prob = 1.0
        else:
            probs = np.exp(np.array(example['token_logprobs'][0]))
            min_prob = np.min(probs)

        if min_prob < threshold:
            low_prob_idxs.append(idx) 
    return low_prob_idxs

def get_low_prob_idxs(data_by_model, gold_idx_list):
    low_idxs_by_model = {}
    for model, data in data_by_model.items():
        if model == "miso":
            low_prob_idxs = get_low_prob(data.items(), is_miso=True)
        else:
            low_prob_idxs = get_low_prob(zip(gold_idx_list,  data), is_miso=False)
        low_idxs_by_model[model] = low_prob_idxs
    return low_idxs_by_model

def get_union(low_idxs_by_model):
    three_way_union = set(low_idxs_by_model['miso']).union(set(low_idxs_by_model['t5'])).union(set(low_idxs_by_model['bart']))
    return three_way_union

def get_line_to_dump(idx, src, tgt, datum_id):
    # split src 
    split_src = re.split("(__User)|(__Agent)|(__StartOfProgram)", src)
    split_src = [x for x in split_src if x != "" and x is not None]
    if len(split_src) == 2:
        user_utt_0 = None
        agent_utt_0 = None
        user_utt_1 = split_src[1].strip()
    elif len(split_src) == 6:
        user_utt_0 = split_src[1].strip()
        agent_utt_0 = split_src[3].strip()
        user_utt_1 = split_src[5].strip()
    else:
        print(split_src)
        print(len(split_src))
        raise AssertionError

    return {"user_turn_0": user_utt_0, "agent_turn_0": agent_utt_0, "user_turn_1": user_utt_1, "tgt": tgt, "idx": idx, "datum_id": datum_id}




if __name__ == "__main__": 
    # paths for calflow test  and valid 
    test_calflow_bart = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20221101T105421.jsonl" 
    test_calflow_t5 = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-large-lm-adapt_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20221102T103315.jsonl"
    test_calflow_miso = "/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/translate_output_calibrated/test_all.tgt"

    test_calflow_models_and_paths = {"miso": test_calflow_miso, "bart": test_calflow_bart, "t5": test_calflow_t5} 

    calflow_gold_path = "/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data.from_benchclamp"

    test_calflow_gold_src_path = f"{calflow_gold_path}/test_all.src_tok"
    test_calflow_gold_tgt_path = f"{calflow_gold_path}/test_all.tgt"
    test_calflow_gold_idx_path = f"{calflow_gold_path}/test_all.idx"
    test_calflow_datum_path = f"{calflow_gold_path}/test.datum_id"

    valid_calflow_gold_src_path = f"{calflow_gold_path}/dev_all.src_tok"
    valid_calflow_gold_tgt_path = f"{calflow_gold_path}/dev_all.tgt"
    valid_calflow_gold_idx_path = f"{calflow_gold_path}/dev_all.idx"
    valid_calflow_datum_path = f"{calflow_gold_path}/dev.datum_id"


    valid_calflow_bart = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_calflow_last_user_all_0.0001_10000_dev_eval_unconstrained-beam_bs_5/model_outputs.20221102T231848.jsonl"
    valid_calflow_t5 = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-large-lm-adapt_calflow_last_user_all_0.0001_10000_dev_eval_unconstrained-beam_bs_5/model_outputs.20221105T115900.jsonl"
    valid_calflow_miso = "/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/translate_output_calibrated/dev_all.tgt"

    valid_calflow_models_and_paths = {"miso": valid_calflow_miso, "bart": valid_calflow_bart, "t5": valid_calflow_t5} 

    # paths for treedst test and valid
    treedst_gold_path = "/brtx/601-nvme1/estengel/resources/data/tree_dst.agent.data"

    test_treedst_bart = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_tree_dst_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20221102T103357.jsonl" 
    test_treedst_t5 = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-large-lm-adapt_tree_dst_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20221106T140554.jsonl" 
    test_treedst_miso = "/brtx/603-nvme1//estengel/calflow_calibration/tree_dst/tune_roberta/translate_output_calibrated/test.tgt"
    test_treedst_models_and_paths = {"miso": test_treedst_miso, "bart": test_treedst_bart, "t5": test_treedst_t5} 

    test_treedst_gold_src_path = f"{treedst_gold_path}/test.src_tok"
    test_treedst_gold_tgt_path = f"{treedst_gold_path}/test.tgt"
    test_treedst_gold_idx_path = f"{treedst_gold_path}/test.idx"
    test_treedst_datum_path = f"{treedst_gold_path}/test.datum_id"

    valid_treedst_bart = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_tree_dst_last_user_all_0.0001_10000_dev_eval_unconstrained-beam_bs_5/model_outputs.20221104T023605.jsonl" 
    # not ready yet 
    valid_treedst_t5 = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-large-lm-adapt_tree_dst_last_user_all_0.0001_10000_dev_eval_unconstrained-beam_bs_5/model_outputs.20221106T135426.jsonl"
    valid_treedst_miso = "/brtx/603-nvme1//estengel/calflow_calibration/tree_dst/tune_roberta/translate_output_calibrated/valid.tgt"
    valid_treedst_models_and_paths = {"miso": test_treedst_miso, "bart": test_treedst_bart, "t5": test_treedst_t5} 

    valid_treedst_gold_src_path = f"{treedst_gold_path}/valid.src_tok"
    valid_treedst_gold_tgt_path = f"{treedst_gold_path}/valid.tgt"
    valid_treedst_gold_idx_path = f"{treedst_gold_path}/valid.idx"
    valid_treedst_datum_path = f"{treedst_gold_path}/valid.datum_id"

    # create a dict for reading 
    arg_dict = {"calflow": {"test": (test_calflow_models_and_paths["miso"], 
                                            test_calflow_models_and_paths["bart"], 
                                            test_calflow_models_and_paths["t5"], 
                                            test_calflow_gold_tgt_path, 
                                            test_calflow_gold_idx_path),
                            "valid": (valid_calflow_models_and_paths["miso"],
                                            valid_calflow_models_and_paths["bart"],      
                                            valid_calflow_models_and_paths["t5"],
                                            valid_calflow_gold_tgt_path,
                                            valid_calflow_gold_idx_path)
                            },
                "treedst": {"test": (test_treedst_models_and_paths["miso"],
                                            test_treedst_models_and_paths["bart"],
                                            test_treedst_models_and_paths["t5"],
                                            test_treedst_gold_tgt_path,
                                            test_treedst_gold_idx_path),
                            "valid": (valid_treedst_models_and_paths["miso"],
                                            valid_treedst_models_and_paths["bart"],
                                            valid_treedst_models_and_paths["t5"],
                                            valid_treedst_gold_tgt_path,
                                            valid_treedst_gold_idx_path)
                            }                             
                }

    print(f"Reading all paths...")
    # read all paths 
    data_and_idx_dict = {"calflow": {"test": None, "valid": None},
                        "treedst": {"test": None, "valid": None}}

    for dataset in ["calflow", "treedst"]:
        for split in ["test", "valid"]:
            miso_path, bart_path, t5_path, gold_path, idx_path = arg_dict[dataset][split]
            print(f"\tReading {dataset} {split}...")
            data_and_idx_dict[dataset][split] = get_data(miso_path, bart_path, t5_path, gold_path, idx_path)

    # get the low prob idxs
    low_prob_idxs_dict =  {"calflow": {"test": None, "valid": None},
                        "treedst": {"test": None, "valid": None}}

    hard_idxs_dict =  {"calflow": {"test": None, "valid": None},
                        "treedst": {"test": None, "valid": None}}

    print("Getting low prob idxs...")
    for dataset in ["calflow", "treedst"]:
        for split in ["test", "valid"]:
            print(f"\tGetting low prob idxs for {dataset} {split}...")
            tgt_by_idx, src_list, tgt_list, idx_list, datum_id_list, data_by_model = data_and_idx_dict[dataset][split]
            low_prob_idxs_dict[dataset][split] = get_low_prob_idxs(tgt_by_idx, idx_list, data_by_model)
            hard_idxs_dict[dataset][split] = get_union(low_prob_idxs_dict[dataset][split])

            print(f"Writing {dataset} {split}...")
            with open(f"../data_subsets/{dataset}/{split}/hard.jsonl", "w") as hard_f,\
                open(f"../data_subsets/{dataset}/{split}/easy.jsonl", "w") as easy_f:
                for idx, src, tgt, datum_id in zip(idx_list, src_list, tgt_list, datum_id_list):
                    to_dump = get_line_to_dump(idx, src, tgt, datum_id)
                    if idx in hard_idxs_dict[dataset][split]:
                        hard_f.write(json.dumps(to_dump) + "\n")
                    else:
                        easy_f.write(json.dumps(to_dump) + "\n")

