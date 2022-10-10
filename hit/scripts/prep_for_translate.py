import argparse
import json
import re 
import pdb 
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
    parser.add_argument("--n_pred", type=int, default=10)
    parser.add_argument("--out_file", type=str, default="hit/data/for_translate/dev_data_by_bin.jsonl")
    args = parser.parse_args()

    split = "dev" if "dev" in args.miso_pred_file else "test"
    # read gold source and target data 
    data_dir = Path(args.data_dir)
    with open(data_dir / f"{split}/{split}_data_by_bin.src_tok", "r") as src_f: 
        gold_src_data = src_f.readlines()
    with open(data_dir / f"{split}/{split}_data_by_bin.tgt", "r") as f:
        gold_tgt_data = [x.strip() for x in f.readlines()]

    # match with gold source 


    tgts = []
    with open(args.miso_pred_file, "r") as f:
        for i, line in enumerate(f):
            line_idx = i // args.n_pred
            src_str = gold_src_data[line_idx]
            prev_user_str, prev_agent_str, user_str = split_source(src_str)
            data = json.loads(line)
            assert(data['src_str'].strip() == re.sub("__StartOfProgram", "", src_str).strip())
            tgt = data["tgt_str"].strip()
            # put into same format as translation data 
            tgt = parse_lispress(tgt)
            tgt = render_compact(tgt)
            tgt = detokenize_lispress(tgt)

            # create dict to write 
            to_write = {"dialogue_id": line_idx, 
                        "turn_part_index": i,
                        "last_agent_utterance": prev_agent_str,
                        "last_user_utterance": prev_user_str, 
                        "utterance": user_str,
                        "plan": tgt}

            tgts.append(to_write)
    with open(args.out_file, "w") as f:
        for tgt in tgts:
            f.write(json.dumps(tgt) + "\n")