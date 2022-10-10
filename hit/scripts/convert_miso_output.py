import argparse
from pathlib import Path
import re 
import json 
from collections import defaultdict
import pdb 
import numpy as np
from dataflow.core.lispress import parse_lispress, render_compact
np.random.seed(12)

def group_by_source(outputs, translated_tgts, n_preds, n_per_ex):
    grouped = defaultdict(list)
    for i in range(0, len(outputs), n_preds):
        lines = outputs[i: i+n_preds]
        translated_lines = translated_tgts[i: i+n_preds]
        # make sure ordered correctly 
        try:
            total_probs = [np.exp(np.sum(np.log(x['expression_probs']))) 
                            if x['expression_probs'] is not None else 0.0 
                                 for x in lines ]
        except:
            pdb.set_trace()
        # sort by total probability 
        combo_lines = zip(lines, translated_lines, total_probs)
        sorted_combo_lines = sorted(combo_lines, key=lambda x: x[2], reverse=True)
        lines, translated_lines, total_probs = zip(*sorted_combo_lines)
        assert(np.argmax(total_probs) == 0)
        for line, tgt in zip(lines, translated_lines):
            # skip these, some small issue with nucleus sampling and target copies
            # causes UNKs for ~2 examples out of ~7000 
            if "UNK" in line['tgt_str']:
                continue
            line['translated'] = tgt
            grouped[i].append(line)
            # add up to n_per_ex examples 
            if len(grouped[i]) == n_per_ex:
                break
    return grouped

def get_distractors(grouped_lines):
    # for each group, choose a random distractor
    distractors = []
    for group in grouped_lines.keys():
        example = grouped_lines[group][0]
        possible_distractors = [x for x in grouped_lines.values() if x[0]['src_str'] != example['src_str']
                                and x[0]['tgt_str'] != example['tgt_str']]
        # flatten
        possible_distractors = [x for y in possible_distractors for x in y]
        distractor = np.random.choice(possible_distractors)
        # distractor = np.random.choice(distractor_group)
        distractors.append(distractor)
    return distractors

def clean_generations(outputs):
    for i in range(len(outputs)):
        outputs[i] = re.sub("<pad>","", outputs[i])
        outputs[i] = re.sub("</?s>", "", outputs[i])
        outputs[i] = re.split("\|", outputs[i])[1].strip()
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--miso_pred_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="hit/data/for_miso")
    parser.add_argument("--translated_tgt_file", type=str, default = "hit/data/translated_by_bart_large/generated_predictions.txt")
    parser.add_argument("--n_preds", type=int, default=10)
    parser.add_argument("--n_per_ex", type=int, default=3)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--filter_fences", action="store_true")
    args = parser.parse_args()

    # get predicted data 
    with open(args.miso_pred_file, "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    # get corresponding gold target data
    data_dir = Path(args.data_dir)
    with open(data_dir / "data_by_bin.src_tok", "r") as src_f: 
        srcs = src_f.readlines()
    with open(data_dir / "data_by_bin.tgt", "r") as f:
        gold_tgt_data = [x.strip() for x in f.readlines()]
    with open(data_dir / "data_by_bin.idx", "r") as f:
        gold_idx_data = [int(x) for x in f.readlines()]
    with open(data_dir / "data_by_bin.bins", "r") as f:
        bin_data = [float(x) for x in f.readlines()]

    # get corresponding translated target data
    with open(args.translated_tgt_file, "r") as f:
        translated_tgt_data = clean_generations(f.readlines()) 

    # break into groups  
    grouped_data = group_by_source(data, translated_tgt_data, args.n_preds, args.n_per_ex)
    # get distractors 
    distractors = get_distractors(grouped_data) 

    output_data = []
    # combine with gold data and distractor data 
    for group_idx, output_list in grouped_data.items():
        try:
            gold_src = srcs[group_idx // args.n_preds]
        except IndexError:
            pdb.set_trace()
        # pdb.set_trace()
        # assert(output_list[0]['src_str'].strip() == gold_src.strip())
        pred_tgts = [render_compact(parse_lispress(x['tgt_str'])) for x in output_list]
        pred_translated = [x['translated'] for x in output_list]
        distractor_tgt = distractors[group_idx // args.n_preds]['translated']
        assert(distractor_tgt not in pred_tgts)
        output_dict = {"gold_tgt": gold_tgt_data[group_idx // args.n_preds],
                        "bin": bin_data[group_idx // args.n_preds],
                        "data_idx": gold_idx_data[group_idx // args.n_preds],
                        "pred_tgts": pred_tgts,
                        "pred_translated": pred_translated,
                        "distractor": distractor_tgt}
        output_data.append(output_dict)

    out_dir = Path(args.out_dir)
    with open(out_dir / "data_for_hit.jsonl", "w") as f1:
        for line in output_data:
            f1.write(json.dumps(line) + "\n")
    # print(len(grouped_data))
    