import argparse
import json
import re 
import pdb 
from collections import defaultdict
from pathlib import Path 
from dataflow.core.lispress import parse_lispress, render_compact

detok_gex = re.compile(r'" (.*?) "')
def detokenize_lispress(lispress):
    lispress = detok_gex.sub(r'"\1"', lispress)
    return lispress

def split_source(input_src_str): 
    input_src_str = input_src_str.strip()
    src_str = re.split("(__User)|(__Agent)|(__StartOfProgram)", input_src_str)
    src_str = [x for x in src_str if x != '' 
                and x is not None 
                and x not in ["__User", "__Agent", "__StartOfProgram"]]
    if len(src_str) == 3:
        # prev user, prev agent, current user 
        return src_str
    else:
        # we only have current user 
        try:
            assert(len(src_str) == 1)
        except:
            pdb.set_trace()
        # prev user, prev agent, current user 
        return "", "", src_str[0]

def read_nucleus_file(miso_pred_file):
    with open(miso_pred_file, "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    to_ret = []
    data_by_idx = defaultdict(list)
    for line in data:
        data_by_idx[line['line_idx']].append(line) 

    for idx, lines in data_by_idx.items():
        total_probs = [np.exp(np.sum(np.log(x['expression_probs']))) 
                                if x['expression_probs'] is not None else 0.0 
                                    for x in lines ]
        min_probs = []
        for x in lines:
            if x['expression_probs'] is not None and len(x['expression_probs']) > 0:
                min_probs.append(np.min(x['expression_probs']))
            else:
                min_probs.append(0.0)

        combo_lines = zip(lines, min_probs, total_probs)
        sorted_combo_lines = sorted(combo_lines, key=lambda x: x[-1], reverse=True)

        data_by_idx[idx] = sorted_combo_lines

    return data_by_idx

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--miso_pred_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="hit/data/for_miso")
    parser.add_argument("--src_file", type=str, default=None)
    parser.add_argument("--tgt_file", type=str, default=None)
    parser.add_argument("--n_pred", type=int, default=10)
    parser.add_argument("--out_file", type=str, default="hit/data/for_translate/dev_data_by_bin.jsonl")
    args = parser.parse_args()

    # get predicted data 
    if args.src_file is None:
        if "dev" not in args.miso_pred_file and "test" not in args.miso_pred_file:
            path = Path(args.miso_pred_file)
            parent = path.parent
            filename = path.stem
            src_file = str(filename + ".src_tok")
            tgt_file = str(filename + ".tgt")
        else:
            split = "dev" if "dev" in args.miso_pred_file else "test"
            src_file = f"{split}/{split}_data_by_bin.src_tok"
            tgt_file = f"{split}/{split}_data_by_bin.tgt" 
    else:
        src_file = args.src_file
        tgt_file = args.tgt_file

    # read gold source and target data 
    data_dir = Path(args.data_dir)
    with open(data_dir / src_file, "r") as src_f: 
        gold_src_data = src_f.readlines()
    with open(data_dir / tgt_file, "r") as f:
        gold_tgt_data = [x.strip() for x in f.readlines()]
    idx_file = tgt_file.split(".tgt")[0] + ".idx"

    with open(data_dir / idx_file) as f:
        gold_idx_data = [int(x) for x in f.readlines()]
    
    src_tgt_by_idx = {idx: (gold, tgt) for idx, gold, tgt in zip(gold_idx_data, gold_src_data, gold_tgt_data)}

    # match with gold source 


    tgts = []
    lines_by_idx = read_nucleus_file(args.miso_pred_file)

    for line_idx, lines in lines_by_idx.items():
        for j, data in enumerate(lines):
            # src_str = gold_src_data[line_idx]
            src_str, __ = src_tgt_by_idx[line_idx]
            prev_user_str, prev_agent_str, user_str = split_source(src_str)
            # data = json.loads(line)
            try:
                assert(data['src_str'].strip() == re.sub("__StartOfProgram", "", src_str).strip())
            except AssertionError:
                pdb.set_trace()
            tgt = data["tgt_str"].strip()
            # put into same format as translation data 
            tgt = parse_lispress(tgt)
            tgt = render_compact(tgt)
            tgt = detokenize_lispress(tgt)

            # create dict to write 
            to_write = {"dialogue_id": line_idx, 
                        "turn_part_index": j,
                        "last_agent_utterance": prev_agent_str,
                        "last_user_utterance": prev_user_str, 
                        "utterance": user_str,
                        "plan": tgt}

            tgts.append(to_write)
    with open(args.out_file, "w") as f:
        for tgt in tgts:
            f.write(json.dumps(tgt) + "\n")