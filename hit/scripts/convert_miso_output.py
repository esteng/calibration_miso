import argparse
from pathlib import Path
import re 
import json 
from collections import defaultdict
from typing import List 
import pdb 
import numpy as np
from dataflow.core.lispress import parse_lispress, render_compact
np.random.seed(12)

def group_by_source(outputs: List, 
                    translated_tgts: List[str], 
                    n_preds: int, 
                    n_per_ex: int,
                    filter_fences: bool):
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
        skip = False
        for line, tgt in zip(lines, translated_lines):
            # skip these, some small issue with nucleus sampling and target copies
            # causes UNKs for ~2 examples out of ~7000 
            if "UNK" in line['tgt_str']:
                continue

            # only skip example if all outputs are fences 
            # since gold fence examples are already skipped 
            # if filter_fences:
                # if "Fence" in line['tgt_str'] or "Pleasantry" in line['tgt_str']:
                    # skip fence examples 
                    # continue 
                    # grouped[i].append(None)
                    # skip = True
                    # continue

            # if not skip:
            line['translated'] = tgt
            grouped[i].append(line)
            # add up to n_per_ex examples 
            if len(grouped[i]) == n_per_ex:
                break
        is_fence = ["Fence" in line['tgt_str'] or "Pleasantry" in line['tgt_str'] for line in grouped[i]]
        if filter_fences and all(is_fence):
            grouped[i] = None

    return grouped

def get_distractors(grouped_lines):
    # for each group, choose a random distractor
    distractors = []
    for group in grouped_lines.keys():
        if grouped_lines[group] is None:
            distractors.append(None)
            continue 

        example = grouped_lines[group][0]
        if example is None:
            distractors.append(None)
            continue
        possible_distractors = [x for x in grouped_lines.values() 
                                if x is not None and x[0] is not None]
        possible_distractors = [x for x in possible_distractors
                                if (x[0]['src_str'] != example['src_str']
                                and x[0]['tgt_str'] != example['tgt_str'])]
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

def get_lispress(entry):
    if entry is None:
        return None
    lispress = parse_lispress(entry['tgt_str'])
    lispress_str = render_compact(lispress)
    return lispress_str


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
    if "dev" not in args.miso_pred_file and "test" not in args.miso_pred_file:

        path = Path(args.miso_pred_file)
        parent = path.parent
        filename = path.stem
        src_file = str(filename + ".src_tok")
        tgt_file = str(filename + ".tgt")
        idx_file = str(filename + ".idx")
        bin_file = str(filename + ".bins")
    else:
        pdb.set_trace()
        split = "dev" if "dev" in args.miso_pred_file else "test"
        src_file = f"{split}/{split}_data_by_bin.src_tok"
        tgt_file = f"{split}/{split}_data_by_bin.tgt" 
        idx_file = f"{split}/{split}_data_by_bin.idx"
        bin_file = f"{split}/{split}_data_by_bin.bins"
    with open(args.miso_pred_file, "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    # get corresponding gold target data
    data_dir = Path(args.data_dir)
    with open(data_dir / src_file, "r") as src_f: 
        srcs = src_f.readlines()
    with open(data_dir / tgt_file, "r") as f:
        gold_tgt_data = [x.strip() for x in f.readlines()]
    with open(data_dir / idx_file, "r") as f:
        gold_idx_data = [int(x) for x in f.readlines()]
    with open(data_dir / bin_file, "r") as f:
        bin_data = [float(x) for x in f.readlines()]

    # get corresponding translated target data
    with open(args.translated_tgt_file, "r") as f:
        translated_tgt_data = clean_generations(f.readlines()) 

    # break into groups  
    grouped_data = group_by_source(data, 
                                    translated_tgt_data, 
                                    args.n_preds, 
                                    args.n_per_ex, 
                                    args.filter_fences)
    # get distractors 
    distractors = get_distractors(grouped_data) 

    output_data = []
    # combine with gold data and distractor data 
    for group_idx, output_list in grouped_data.items():
        try:
            gold_src = srcs[group_idx // args.n_preds]
            gold_tgt = gold_tgt_data[group_idx // args.n_preds]
        except IndexError:
            pdb.set_trace()
        if args.filter_fences and ("Fence" in gold_tgt or "Pleasantry" in gold_tgt):
            # skip fence examples 
            continue
        if args.filter_fences and output_list is None:
            # skip examples where all outputs are fences
            continue 

        if len(output_list) == 1 and output_list[0] is None:
            # skip fence examples 
            continue 

        pred_tgts = [get_lispress(x) for x in output_list]
        pred_translated = [x['translated'] if x is not None else None for x in output_list]
        try:
            distractor = distractors[group_idx // args.n_preds]
            if distractor is None:
                continue
            distractor_tgt = distractor['translated']

        except IndexError:
            pdb.set_trace()
        assert(distractor_tgt not in pred_tgts)
        output_dict = {"gold_tgt": gold_tgt,
                       "gold_src": gold_src,
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
    