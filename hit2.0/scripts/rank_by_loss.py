import json 
import numpy as np 
import argparse
from pathlib import Path
from collections import defaultdict 
import pdb 

def read_loss_file(loss_file):
    with open(loss_file) as f1:
        loss_data = json.load(f1)
    loss_data = [x[0] for x in loss_data]
    return loss_data

def get_seq_prob(loss_row): 
    total_log_prob = 0
    for timestep in loss_row:
        log_prob = np.log(timestep['prob_gold'])
        total_log_prob += log_prob
        
    total_prob = np.exp(total_log_prob) 
    # we don't need to take the average since we're doing a forced decode
    # of the predicted parse for each input, so the output length is 
    # always the same across different inputs 
    return total_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_file", type=str, required=True)
    parser.add_argument("--translated_file", type=str, required=True)
    parser.add_argument("--gold_dir", type=str, required=True)
    parser.add_argument("--gold_split", type=str, default='dev_all')
    parser.add_argument("--force_decode_input_dir", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    loss_data = read_loss_file(args.loss_file)

    miso_dir = Path(args.force_decode_input_dir)
    with open(miso_dir /  f"dev_for_forced_decode.src_tok")  as src_f, \
        open(miso_dir / f"dev_for_forced_decode.tgt") as tgt_f, \
        open(miso_dir / f"dev_for_forced_decode.idx") as idx_f:
        src_data = src_f.readlines()
        tgt_data = tgt_f.readlines()
        idx_data = idx_f.readlines()

    gold_dir = Path(args.gold_dir)
    with open(gold_dir / f"{args.gold_split}.src_tok") as src_f, \
        open(gold_dir / f"{args.gold_split}.tgt") as tgt_f, \
        open(gold_dir / f"{args.gold_split}.idx") as idx_f:
        gold_src_data = src_f.readlines()
        gold_tgt_data = tgt_f.readlines()
        gold_idx_data = idx_f.readlines()
        gold_src_tgt_by_idx = {int(idx): (src,tgt) for idx, src, tgt in zip(gold_idx_data, gold_src_data, gold_tgt_data)}

    src_tgt_loss_by_idx = defaultdict(list)
    for i in range(len(src_data)):
        src_line = src_data[i].strip()
        tgt_line = tgt_data[i].strip()
        idx = int(idx_data[i].strip()) 
        loss_row = loss_data[i]
        seq_prob = get_seq_prob(loss_row)
        src_tgt_loss_by_idx[idx].append((src_line, tgt_line, seq_prob))


    to_ret = []    
    with open(args.out_file, "w") as f1:
        for idx, data in src_tgt_loss_by_idx.items():
            data = sorted(data, key=lambda x: x[2], reverse=True)
            pred_src, pred_tgt, prob = data[0]
            data_to_write = {"pred_src": pred_src, "pred_tgt": pred_tgt, "prob": prob}
            gold_src, gold_tgt = gold_src_tgt_by_idx[idx]
            data_to_write["gold_src"] = gold_src
            data_to_write["gold_tgt"] = gold_tgt

            f1.write(json.dumps(data_to_write) + "\n") 

