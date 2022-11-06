import json
import pdb 
import argparse 
from tqdm import tqdm
from collections import defaultdict
import numpy as np 
import re 
from dataflow.core.lispress import parse_lispress, render_compact, render_pretty
from dataflow.core.linearize import lispress_to_seq

def read_nucleus_file(miso_pred_file):
    with open(miso_pred_file, "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    to_ret = []
    data_by_idx = defaultdict(list)
    data_by_src_str = defaultdict(list)
    for line in data:
        data_by_src_str[line['src_str']].append(line) 
        data_by_idx[line['line_idx']].append(line) 

    for src_str, lines in data_by_src_str.items():
        total_probs = [np.exp(np.sum(np.log(x['expression_probs']))) 
                                if x['expression_probs'] is not None else 0.0 
                                    for x in lines ]
        mean_probs = [np.mean(x['expression_probs']) 
                                if x['expression_probs'] is not None and np.sum(x['expression_probs']) > 0.0 
                                else 0.0 for x in lines ]
        min_probs = []
        for x in lines:
            if x['expression_probs'] is not None and len(x['expression_probs']) > 0:
                min_probs.append(np.min(x['expression_probs']))
            else:
                min_probs.append(0.0)

        combo_lines = zip(lines, min_probs, mean_probs, total_probs)
        sorted_combo_lines = sorted(combo_lines, key=lambda x: x[-1], reverse=True)

        data_by_src_str[src_str] = sorted_combo_lines
        idx = lines[0]['line_idx']
        data_by_idx[idx] = sorted_combo_lines
    return data_by_src_str, data_by_idx

def read_gold_file(file):
    with open(file) as f:
        if file.endswith(".tgt"):
            to_ret = [render_compact(parse_lispress(line)) for line in f.readlines()]
        else:
            to_ret = [re.sub("__StartOfProgram", "", x).strip() for x in f.readlines()]
    return to_ret 


def get_low_prob(iterator, is_miso = False, threshold = 0.6):
    low_prob_idxs = []
    for idx, example in iterator:
        if is_miso: 
            try:
                min_prob = example[0][1]
            except:
                min_prob = np.min(example[0]['expression_probs'])
        else:
            probs = np.exp(np.array(example['token_logprobs'][0]))
            min_prob = np.min(probs)

        if min_prob < threshold:
            low_prob_idxs.append(idx) 
    return low_prob_idxs

def get_combined_acc(bart_outputs, beam_outputs, gold_tgt_by_idx, miso_idxs):
    all_pred_gold_pairs = []

    for idx, gold_tgt in gold_tgt_by_idx.items(): 
        # if confident take miso 
        if str(idx) not in miso_idxs:
            model_name = "miso"
            try:
                pred_str = beam_outputs[idx]
            except KeyError:
                pred_str = beam_outputs[idx]
            gold_str = gold_tgt 
        else:
            model_name = "bart"
            try:
                bart_example = bart_outputs[idx]
            except KeyError:
                pdb.set_trace() 

            pred_str = bart_example['outputs'][0]
            gold_str = bart_example['test_datum_canonical']
        try:
            pred_tgt = render_compact(parse_lispress(pred_str))
        except (AssertionError, IndexError) as e:
            pred_tgt = "(Error)"

        gold_tgt = render_compact(parse_lispress(gold_str))
        all_pred_gold_pairs.append((pred_tgt, gold_tgt, model_name))

    # get accuracy
    correct = 0
    for pred, gold, model_name in all_pred_gold_pairs:
        if pred == gold:
            correct += 1
    return correct / len(all_pred_gold_pairs), all_pred_gold_pairs

if __name__ == "__main__": 
    gold_path = "/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data.from_benchclamp"
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None) 
    args = parser.parse_args()
    if args.threshold is None:
        dev_calflow_bart = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_calflow_last_user_all_0.0001_10000_dev_eval_unconstrained-beam_bs_5/model_outputs.20221102T231848.jsonl" 
        dev_calflow_miso_nuc = "/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/translate_output_calibrated/dev_all.tgt"
        dev_calflow_miso_beam = "/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/translate_output/dev_all.tgt" 


        print("Read gold")
        dev_gold_src = read_gold_file(f"{gold_path}/dev_all.src_tok")
        dev_gold_tgt = read_gold_file(f"{gold_path}/dev_all.tgt")
        dev_gold_idx = read_gold_file(f"{gold_path}/dev_all.idx")
        dev_gold_tgt_by_idx = {idx: gold for idx, gold in zip(dev_gold_idx, dev_gold_tgt)}

        # first use dev data to get a threshold 
        print("read miso")
        with open(dev_calflow_miso_beam) as f:
            dev_miso_beam = [x.strip() for x in f.readlines()]
        dev_miso_beam_by_idx = {idx: beam for idx, beam  in zip(dev_gold_idx, dev_miso_beam)}

        __, dev_miso_data = read_nucleus_file(dev_calflow_miso_nuc)
        print("read bart")
        with open(dev_calflow_bart) as f1:
            dev_bart_data = {str(i): json.loads(x) for i, x in enumerate(f1.readlines())}

        print("compute ")
        thresholds = [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
        accs = []
        for threshold in tqdm(thresholds): 
            dev_miso_low_prob_idxs = get_low_prob(dev_miso_data.items(), is_miso=True, threshold=threshold)

            acc, __ = get_combined_acc(dev_bart_data, 
                                        dev_miso_beam_by_idx, 
                                        dev_gold_tgt_by_idx, 
                                        dev_miso_low_prob_idxs)

            print(f"thresh: {threshold}, acc: {acc}")
            accs.append(acc)
        
        # get the threshold that gives the best accuracy
        best_acc_idx = np.argmax(accs)
        print(f"best acc {accs[best_acc_idx]*100:.1f} at {thresholds[best_acc_idx]}")
        best_threshold = thresholds[best_acc_idx]
    else:
        best_threshold = args.threshold
    # reading test 
    print("reading test data")
    test_calflow_bart = "/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20221101T105421.jsonl" 
    test_calflow_miso = "/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/translate_output_calibrated/test_all.tgt"
    test_calflow_miso_beam = "/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/translate_output/test_all.tgt" 

    test_gold_src = read_gold_file(f"{gold_path}/test_all.src_tok")
    test_gold_tgt = read_gold_file(f"{gold_path}/test_all.tgt")
    test_gold_idx = read_gold_file(f"{gold_path}/test_all.idx")
    test_gold_tgt_by_idx = {idx: gold for idx, gold in zip(test_gold_idx, test_gold_tgt)}

    print("read miso")
    with open(test_calflow_miso_beam) as f:
        test_miso_beam = [x.strip() for x in f.readlines()]
    test_miso_beam_by_idx = {idx: beam for idx, beam  in zip(test_gold_idx, test_miso_beam)}

    __, test_miso_data = read_nucleus_file(test_calflow_miso)
    print("read bart")
    with open(test_calflow_bart) as f1:
        test_bart_data = {str(i): json.loads(x) for i, x in enumerate(f1.readlines())}

    # get indices 
    print("getting indices")
    test_miso_low_prob_idxs = get_low_prob(test_miso_data.items(), is_miso=True, threshold=best_threshold) 
    print("getting acc")
    test_acc, test_preds = get_combined_acc(test_bart_data, test_miso_beam_by_idx, test_gold_tgt_by_idx, test_miso_low_prob_idxs)
    print(f"Ensemble accuracy: {test_acc*100:.1f}")


    with open("analysis/ensemble_data/test_pred.tgt", "w") as pf, open("analysis/ensemble_data/test_gold.tgt", "w") as tf:
        for p,g, m in test_preds:
            pf.write(f"{p}\n")
            tf.write(f"{g}\n")

    # for p,g, m in test_preds:
    #     if p != g:
    #         print(p)
    #         print(g)
    #         print()
    #         pdb.set_trace()