import pathlib
import json
import argparse 
import pandas as pd
from tqdm import tqdm
import pdb 

from dataflow.core.lispress import parse_lispress, lispress_to_program
from dataflow.core.program import ValueOp

def read_tgt_file(path):
    with open(path) as f1:
        return [lispress_to_program(parse_lispress(x), 0)[0] for x in f1.readlines()]

def read_src_strs(path):
    with open(path) as f1:
        src_strs = [x.strip().split(" ") for x in f1.readlines()]
    return src_strs 

def get_source_copies(src_str, program):
    possible_strings = []
    for i, expr in enumerate(program.expressions):
        op = expr.op
        if isinstance(op, ValueOp):
            value = json.loads(op.value)
            possible_strings.append(str(value['underlying']).strip())

    all_span_lens = set([len(x.strip().split(" ")) for x in possible_strings])
    src_str_set = []
    for span_len in all_span_lens:
        for idx in range(0, len(src_str)-span_len + 1, 1): 
            src_str_set.append(" ".join(src_str[idx: idx+span_len]))

    string_copies = [x for x in possible_strings if x in src_str_set]
    return string_copies 

def source_copy_metric(src_str, pred_program, gold_program):
    gold_source_copies = get_source_copies(src_str, gold_program)
    pred_source_copies = get_source_copies(src_str, pred_program)

    if len(gold_source_copies) == 0:
        return None, None, None

    true_positives = sum([1 for x in pred_source_copies if x in gold_source_copies])
    false_negatives = sum([1 for x in gold_source_copies if x not in pred_source_copies])
    false_positives = sum([1 for x in pred_source_copies if x not in gold_source_copies])

    try:
        precision = true_positives/(true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    try: 
        recall = true_positives/(true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return precision, recall, f1

def main(src_path, pred_tgt_path, gold_tgt_path):
    gold_programs = read_tgt_file(gold_tgt_path)
    pred_programs = read_tgt_file(pred_tgt_path)
    src_strs = read_src_strs(src_path) 
    no_copy = 0
    df = pd.DataFrame(columns=[ "metric", "value"], dtype=object)
    for i, (src_str, pred_program, gold_program) in tqdm(enumerate(zip(src_strs, pred_programs, gold_programs))):
        p, r, f1 = source_copy_metric(src_str, pred_program, gold_program)
        if p is None:
            no_copy += 1
        df = df.append({"metric": "precision", "value": p}, ignore_index=True)
        df = df.append({"metric": "recall", "value": r}, ignore_index=True)
        df = df.append({"metric": "f1", "value": f1}, ignore_index=True)

    print(f"Precision: {df[df['metric'] == 'precision'].mean()}") 
    print(f"Recall: {df[df['metric'] == 'recall'].mean()}") 
    print(f"F1: {df[df['metric'] == 'f1'].mean()}") 

    print(f"for path {src_path} there are {no_copy} examples without copy: {no_copy/len(src_strs) * 100:.2f}")

    return df 