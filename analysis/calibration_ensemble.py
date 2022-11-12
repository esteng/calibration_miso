import json
import pdb 
import argparse 
from tqdm import tqdm
from collections import defaultdict
import numpy as np 
import re 
from dataflow.core.lispress import parse_lispress, render_compact, render_pretty
from dataflow.core.linearize import lispress_to_seq
from calibration_metric.metric import ECEMetric
from calibration_utils import single_exact_match, read_nucleus_file, read_gold_file

def get_low_prob(iterator, is_miso = False, threshold = 0.6):
    low_prob_idxs = []
    probs_by_idx = {}
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
        probs_by_idx[idx] = min_prob
    return low_prob_idxs, probs_by_idx

def get_combined_acc(bart_outputs, beam_outputs, gold_tgt_by_idx, miso_idxs, miso_probs_by_idx):
    all_pred_gold_pairs = []
    all_probs = []
    for idx, gold_tgt in gold_tgt_by_idx.items(): 
        # if confident take miso 
        if str(idx) not in miso_idxs:
            prob = miso_probs_by_idx[idx]
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

            prob = np.min(np.exp(np.array(bart_example['token_logprobs'][0])))
            pred_str = bart_example['outputs'][0]
            gold_str = bart_example['test_datum_canonical']
        try:
            pred_tgt = render_compact(parse_lispress(pred_str))
        except (AssertionError, IndexError) as e:
            pred_tgt = "(Error)"

        gold_tgt = render_compact(parse_lispress(gold_str))
        all_pred_gold_pairs.append((pred_tgt, gold_tgt, model_name))
        all_probs.append(prob)
    # get accuracy
    correct = 0
    for pred, gold, model_name in all_pred_gold_pairs:
        match, __ = single_exact_match(pred, gold)
        if match:
            correct += 1
    return correct / len(all_pred_gold_pairs), all_pred_gold_pairs, all_probs

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
    test_miso_low_prob_idxs, test_miso_probs_by_idx = get_low_prob(test_miso_data.items(), is_miso=True, threshold=best_threshold) 
    print(f"MISO has {len(test_miso_low_prob_idxs)}")
    print("getting acc")
    test_acc, test_preds, test_probs = get_combined_acc(test_bart_data, 
                                            test_miso_beam_by_idx, 
                                            test_gold_tgt_by_idx, 
                                            test_miso_low_prob_idxs, 
                                            test_miso_probs_by_idx)
    print(f"Total: {len(test_preds)}")
    print(f"Ensemble accuracy: {test_acc*100:.1f}")

    test_accs = []
    for pred, gold, model_name in test_preds:
        match, __ = single_exact_match(pred, gold)
        if match:
            test_accs.append(1)
        else:
            test_accs.append(0)
    ece_score = ECEMetric(n_bins=10)(np.array(test_probs), np.array(test_accs))
    print(f"ECE: {ece_score*100:.2f}")
    
    # get all bart_probs:
    bart_probs = []
    bart_accs = []
    for bart_example in test_bart_data.values():
        prob = np.min(np.exp(np.array(bart_example['token_logprobs'][0])))
        pred_str = bart_example['outputs'][0]
        gold_str = bart_example['test_datum_canonical']
        match, __ = single_exact_match(pred_str, gold_str)
        if match:
            bart_accs.append(1)
        else:
            bart_accs.append(0)
        bart_probs.append(prob)

        
    ece_score = ECEMetric(n_bins=10)(np.array(bart_probs), np.array(bart_accs))
    print(f"BART ECE: {ece_score*100:.2f}")

    miso_probs, miso_accs = [], []
    skipped = 0
    # get all miso_probs:
    for idx, gold_tgt in test_gold_tgt_by_idx.items():
        try:
            prob = test_miso_probs_by_idx[idx]
            pred_str = test_miso_beam_by_idx[idx]
        except KeyError:
            skipped += 1
            continue
        gold_str = gold_tgt 
        match, __ = single_exact_match(pred_str, gold_str)
        if match:
            miso_accs.append(1)
        else:
            miso_accs.append(0)
        miso_probs.append(prob)

    print(f'skipped {skipped} examples')
    ece_score = ECEMetric(n_bins=10)(np.array(miso_probs), np.array(miso_accs))
    print(f"MISO ECE: {ece_score*100:.2f}")

    # with open("analysis/ensemble_data/test_pred.tgt", "w") as pf, open("analysis/ensemble_data/test_gold.tgt", "w") as tf:
        # for p,g, m in test_preds:
            # pf.write(f"{p}\n")
            # tf.write(f"{g}\n")


