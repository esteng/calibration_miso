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

    # match with gold source 


    tgts = []
    lines_by_idx = defaultdict(list)
    with open(args.miso_pred_file, "r") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            lines_by_idx[line['line_idx']].append(line)

    for line_idx, lines in lines_by_idx.items():
        for j, data in enumerate(lines):
            # line_idx = i // args.n_pred
            line_idx = int(line_idx)
            src_str = gold_src_data[line_idx]
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