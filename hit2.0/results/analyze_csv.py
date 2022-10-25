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

def score_lines(csv_lines, json_lines):
    true_positives = []
    false_positives = []
    false_negatives = []
    for cl, jl in zip(csv_lines, json_lines):
        accept = cl['Answer.radio-input'] == "yes"
        prob = float(jl['prob'])
        was_correct = jl['pred_tgt'].strip() == jl['gold_tgt'].strip()

        if accept and was_correct:
            true_positives.append(jl)
        if not accept and was_correct:
            false_negatives.append(jl)
        if accept and not was_correct:
            false_positives.append(jl)

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1, true_positives, false_positives, false_negatives


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()

    csv_lines = read_csv(args.csv)
    json_lines = read_json(args.json)

    assert len(csv_lines) == len(json_lines)

    (precision_score,
    recall_score,
    f1_score,
    tp_list,
    fp_list,
    fn_list) = score_lines(csv_lines, json_lines)

    print(f"{precision_score:.3f} {recall_score:.3f} {f1_score:.3f}")
    pdb.set_trace()