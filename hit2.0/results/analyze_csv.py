import argparse
from collections import defaultdict, Counter
import csv 
import pathlib
import json
import re 
import os 
import pdb 
import subprocess 
import numpy as np

def read_csv(csv_file):
    with open(csv_file) as f1:
        reader = csv.DictReader(f1)
        csv_lines = [x for x in reader]
    return csv_lines

def read_json(path):
    with open(path) as f1:
        data = [json.loads(x) for x in f1.readlines()]
    return data

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator

def score_lines(csv_lines, json_lines, bins_by_idx):
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []

    all_bins = set(bins_by_idx.values())
    by_bin_data = {bin: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for bin in all_bins}

    for cl, jl in zip(csv_lines, json_lines):
        accept = cl['Answer.radio-input'] == "yes"
        prob = float(jl['prob'])
        was_correct = jl['pred_tgt'].strip() == jl['gold_tgt'].strip()
        bin = bins_by_idx[jl['idx']]
        
        if accept and was_correct:
            true_positives.append(jl)
            by_bin_data[bin]['tp'] += 1
        if not accept and was_correct:
            false_negatives.append(jl)
            by_bin_data[bin]['fn'] += 1
        if accept and not was_correct:
            false_positives.append(jl)
            by_bin_data[bin]['fp'] += 1
        if not accept and not was_correct:
            true_negatives.append(jl)
            by_bin_data[bin]['tn'] += 1
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = 2 * safe_divide(precision * recall, precision + recall)

    for bin, data in by_bin_data.items():
        btp = data['tp']
        bfp = data['fp']
        bfn = data['fn']
        tn = data['tn']
        bin_precision = safe_divide(btp, btp + bfp)
        bin_recall = safe_divide(btp, btp + bfn)
        bin_f1 = 2 * safe_divide(bin_precision * bin_recall, bin_precision + bin_recall)
        by_bin_data[bin]['precision'] = bin_precision
        by_bin_data[bin]['recall'] = bin_recall
        by_bin_data[bin]['f1'] = bin_f1
        
    to_ret = {"total_precision": precision,
              "total_recall": recall,
              "total_f1": f1,
              "by_bin_data": by_bin_data,
              "true_positives": true_positives,
              "false_positives": false_positives,
              "false_negatives": false_negatives,
              "true_negatives": true_negatives}
    return to_ret

def get_bins(translation_dir):
    translation_dir = pathlib.Path(translation_dir)
    bin_file = translation_dir / "sampled_bins.txt"
    idx_file = translation_dir / "sampled_idxs.txt"
    with open(bin_file) as f1:
        bins = [float(x.strip()) for x in f1.readlines()]
    with open(idx_file) as f1:
        idxs = [int(x.strip()) for x in f1.readlines()]
    bins_by_idx = {idx: bin for bin, idx in zip(bins, idxs)}
    return bins_by_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--translation_dir", type=str, required=True, help="something like hit2.0/data/for_translate_pilot")
    args = parser.parse_args()

    csv_lines = read_csv(args.csv)
    json_lines = read_json(args.json)

    bins_by_idx = get_bins(args.translation_dir)

    assert len(csv_lines) == len(json_lines)

    score_results = score_lines(csv_lines, json_lines, bins_by_idx)
    precision_score = score_results['total_precision']
    recall_score = score_results['total_recall']
    f1_score = score_results['total_f1']
    print(f"{precision_score:.3f} {recall_score:.3f} {f1_score:.3f}")

    for bin, res_dict in score_results['by_bin_data'].items():
        print(f"\tbin: {bin:.2f}, p: {res_dict['precision']:.3f}, r: {res_dict['recall']:.3f}, f1: {res_dict['f1']:.3f}")
    pdb.set_trace()