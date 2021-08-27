import argparse
import pathlib 
import json
import pdb 
from collections import defaultdict

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--train",type=str, required=True)
    parser.add_argument("--fxn-of-interest",type=str, default="FindManager")
    parser.add_argument("--annotation-path", type=str, required=True, help="path to annotated minimal pairs, output of running a model over manually modified minimal pair inputs")
    parser.add_argument("--fxn-lines", type=str, required=True)
    parser.add_argument("--out",type=str, required=True)
    args = parser.parse_args() 

    train_src = args.train + ".src_tok"
    train_tgt = args.train + ".tgt"
    train_idxs = args.train + ".idx"
    with open(train_src) as src, open(train_tgt) as tgt:
        src_lines = src.readlines()
        tgt_lines = tgt.readlines()

    annotation_src = args.annotation_path + ".src_tok"
    annotation_tgt = args.annotation_path + ".tgt"
    with open(annotation_src) as src, open(annotation_tgt) as tgt:
        annotation_src_lines = src.readlines()
        annotation_tgt_lines = tgt.readlines()

    fxn_lines = [line.strip() for line in open(args.fxn_lines).readlines()]

    lookup_table = {}
    with open(train_idxs, "w") as f1:
        c = 0
        for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
            tgt_split = tgt_line.split(" ")
            if args.fxn_of_interest not in tgt_split:
                continue
            
            idx = fxn_lines.index(tgt_line.strip())

            new_src = annotation_src_lines[idx]
            new_tgt = annotation_tgt_lines[idx]
            c += 1
            if args.fxn_of_interest in tgt_line:
                lookup_table[i] = (new_src.strip(), new_tgt.strip())
                # llookup_table[i] = (src_line.strip(), tgt_line.strip())
            f1.write(f"{i}\n")
    with open(args.out,"w") as f1:
        json.dump(lookup_table, f1)