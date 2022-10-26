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
from dataflow.core.lispress import parse_lispress, render_compact

def clean_lispress(x):
    x_before = x
    x = x.strip()
    # if False:
    # remove determiners in value ops 
    x = re.sub('" (a|an|the) ([^"]+) "', '" \g<2> "', x)
    # lowercase all value ops 
    x = re.sub('" ([^"]+) "', lambda m: m.group(0).lower(), x)
    # if x != x_before:
    # pdb.set_trace() 

    parsed_x = parse_lispress(x)
    return render_compact(parsed_x)

def decode(rewritten, checkpoint_dir):
    # write everything to a temp file 
    tempfile_src = pathlib.Path(__file__).parent / "temp.src_tok"
    tempfile_tgt = pathlib.Path(__file__).parent / "temp.tgt"
    tempfile_idx = pathlib.Path(__file__).parent / "temp.idx"
    with open(tempfile_src,"w") as f1, \
        open(tempfile_tgt,"w") as f2, \
        open(tempfile_idx,"w") as f3:
        for src, tgt, idx in rewritten:
            f1.write(src + "\n")
            f2.write(tgt + "\n")
            f3.write(str(idx).strip() + "\n")

    # run the decoding script
    decode_command = ["sh", "/home/estengel/incremental-function-learning/experiments/calflow.sh", "-a", "eval_fxn"]
    env = os.environ.copy()
    # print(temp_file)
    env['CHECKPOINT_DIR'] = checkpoint_dir
    env['TEST_DATA'] = str(tempfile_src.parent / "temp")
    env['FXN'] = "none"
    p = subprocess.Popen(decode_command, stdout=subprocess.PIPE, env=env)
    out, errs = p.communicate()
    # print(out)
    # print(errs)
    # out = out.decode('utf-8')
    # errs = errs.decode('utf-8')
    out_file = pathlib.Path(checkpoint_dir) / "translate_output" / f"temp.tgt"
    out_lispress = []
    with open(out_file) as f1:
        for l in f1.readlines():
            lispress = clean_lispress(l)
            out_lispress.append(lispress)
    return out_lispress

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

def build_miso_input(csv_line):
    user_context = csv_line['Input.user_turn_0'].strip()
    agent_context = csv_line['Input.agent_turn_0'].strip()
    user_input = csv_line['Input.user_turn_1'].strip()
    inp_str = ""
    if user_context != "":
        inp_str += f"__User {user_context} __Agent {agent_context} "
    inp_str += f"__User {user_input} __StartOfProgram"
    return inp_str

def score_lines(csv_lines, json_lines, bins_by_idx, rewrite=False, rewrite_checkpoint = None):
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []

    all_bins = set(bins_by_idx.values())
    by_bin_data = {bin: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for bin in all_bins}

    if rewrite:
        rewritten = []
        # build input based on gloss 
        for cl, jl in zip(csv_lines, json_lines):
            miso_input = build_miso_input(cl)
            miso_output = jl['gold_tgt'].strip()
            rewritten.append((miso_input, miso_output, jl['idx']))
        # replace json tgts with predicted from the gloss 
        print(f"Decoding from {rewrite_checkpoint}...")
        pred_tgts = decode(rewritten, rewrite_checkpoint)
        for i, jl in enumerate(json_lines):
            # pdb.set_trace()
            pred_tgt = pred_tgts[i]
            jl['pred_tgt'] = pred_tgt
            json_lines[i] = jl

    for cl, jl in zip(csv_lines, json_lines):
        accept = cl['Answer.radio-input'] == "yes"
        prob = float(jl['prob'])
        was_correct = clean_lispress(jl['pred_tgt'].strip()) == clean_lispress(jl['gold_tgt'].strip())
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

def run_one_baseline(json_lines, bins_by_idx, cutoff):
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []

    all_bins = set(bins_by_idx.values())
    by_bin_data = {bin: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for bin in all_bins}

    for jl in json_lines:
        # here's the difference: 
        accept = jl['confidence'] >= cutoff 
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

def run_baseline(json_lines, bins_by_idx, max_bin):
    all_results = []
    increment = max_bin / 10
    all_cutoffs = np.arange(increment, max_bin, increment)
    for cutoff in all_cutoffs:
        all_results.append((cutoff, run_one_baseline(json_lines, bins_by_idx, cutoff)))
    all_f1s = [result['total_f1'] for cutoff, result in all_results]
    best_idx = np.argmax(all_f1s)
    best_cutoff, best_result = all_results[best_idx]
    return best_cutoff, best_result
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--translation_dir", type=str, required=True, help="something like hit2.0/data/for_translate_pilot")
    parser.add_argument("--do_baseline", action="store_true")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--rewrite_checkpoint", type=str, default=None)
    parser.add_argument("--max_bin", type=float, default=0.6)
    args = parser.parse_args()

    csv_lines = read_csv(args.csv)
    json_lines = read_json(args.json)

    bins_by_idx = get_bins(args.translation_dir)
    if not args.do_baseline:
        assert len(csv_lines) == len(json_lines)
        score_results = score_lines(csv_lines, json_lines, bins_by_idx, args.rewrite, args.rewrite_checkpoint)
    else:
        print("RUNNING BASELINE...")
        # baseline: just set a confidence cutoff and reject everything below it 
        # we will run over a range of cutoffs 
        best_cutoff, score_results = run_baseline(json_lines, bins_by_idx, args.max_bin)
        print(f"Best cutoff: {best_cutoff}")

    precision_score = score_results['total_precision']
    recall_score = score_results['total_recall']
    f1_score = score_results['total_f1']
    print(f"{precision_score:.3f} {recall_score:.3f} {f1_score:.3f}")

    for bin, res_dict in sorted(score_results['by_bin_data'].items(), key = lambda x: x[0]):
        print(f"\tbin: {bin:.2f}, p: {res_dict['precision']:.3f}, r: {res_dict['recall']:.3f}, f1: {res_dict['f1']:.3f}")
    pdb.set_trace()