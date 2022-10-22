import json
import re 
import pdb
from pathlib import Path
import argparse

from dataflow.core.utterance_tokenizer import UtteranceTokenizer
from dataflow.core.lispress import parse_lispress, render_compact
from dataflow.core.linearize import lispress_to_seq

from prep_for_translate import read_nucleus_file

tokenizer = UtteranceTokenizer()

def replace_src(src_str, new_user_str):
    split_str = re.split("(__User)|(__Agent)", src_str) 
    split_str = [x.strip() for x in split_str if x not in [None, '']]
    split_str[-1] = new_user_str
    split_str.append("__StartOfProgram")
    return " ".join(split_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--miso_pred_file", default=None)
    parser.add_argument("--translated_file", default=None)
    parser.add_argument("--n_preds", type=int, default=10)
    parser.add_argument("--out_dir", default="hit2.0/data/for_forced_decode")
    args = parser.parse_args()

    # read miso_pred_tgts and glosses 
    nucleus_lines = read_nucleus_file(args.miso_pred_file)

    pred_src_data = []
    with open(args.translated_file, "r") as f:
        for x in f.readlines():
            x = re.sub("<.*?>","",x)
            x = re.sub("\|", "", x)
            x = x.strip()
            pred_src_data.append(x)

    # chunk translated data into chunks of n_preds
    pred_src_data_chunks = []
    for i in range(0, len(pred_src_data), args.n_preds):
        pred_src_data_chunks.append(pred_src_data[i:i+args.n_preds])

    # remove duplicates
    for i in range(len(pred_src_data_chunks)):
        pred_src_data_chunks[i] = list(set(pred_src_data_chunks[i]))
    print(len(pred_src_data_chunks))
    print(len(nucleus_lines))

    out_dir = Path(args.out_dir)
    to_write_src, to_write_tgt, to_write_idx = [], [], []
    for i, (idx, nuc_data) in enumerate(nucleus_lines.items()):
        nuc_tgt = nuc_data[0][0]['tgt_str']
        nuc_tgt = lispress_to_seq(parse_lispress(nuc_tgt))
        nuc_tgt = " ".join(nuc_tgt)

        gold_src  = nuc_data[0][0]['src_str']
        pred_src_options = pred_src_data_chunks[i]
        for pred_src in pred_src_options:
            pred_src = tokenizer.tokenize(pred_src)
            pred_src = " ".join(pred_src)
            combo_src = replace_src(gold_src, pred_src)
            to_write_src.append(combo_src)
            to_write_tgt.append(nuc_tgt)
            to_write_idx.append(idx)

    with open(out_dir /  f"dev_for_forced_decode.src_tok", "w") as src_f, \
        open(out_dir / f"dev_for_forced_decode.tgt", "w") as tgt_f, \
        open(out_dir / f"dev_for_forced_decode.idx", "w") as idx_f:
        for src, tgt, idx in zip(to_write_src, to_write_tgt, to_write_idx):
            src_f.write(src + "\n")
            tgt_f.write(tgt + "\n") 
            idx_f.write(str(idx) + "\n") 
