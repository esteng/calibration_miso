import json 
import argparse 
import pathlib 
import pdb 
import re

import numpy as np 

# compute rank of manually constructed min pairs 
def compute_idx_rank(generated_idxs, true_idx):
    return generated_idxs.index(true_idx)

def get_avg_rank(path, k=3):
    with open(path.joinpath("generated_lookup_table.json")) as f1:
        gen_table = json.load(f1)
    with open(path.joinpath("lookup_table.json")) as f1:
        manual_table = json.load(f1)
    
    with open(path.joinpath("train.src_tok")) as src_f, open(path.joinpath("train.idx")) as idx_f:
        train_src = [line.strip() for line in src_f.readlines()]
        train_idxs = [int(line.strip()) for line in idx_f.readlines()]

    with open(path.joinpath("fxn_train.src_tok")) as src_f, open(path.joinpath("fxn_train.idx")) as idx_f:
        fxn_src = [line.strip() for line in src_f.readlines()]
        fxn_idxs = [int(line.strip()) for line in idx_f.readlines()]

    ranks = []
    for i, idx in enumerate(fxn_idxs): 
        manual_example_src, __ = manual_table[str(idx)]
        manual_example_idx = train_src[-100:].index(manual_example_src)
        ranked_examples = gen_table[str(idx)]
        rank = ranked_examples.index(manual_example_idx)
        ranks.append(rank/len(ranked_examples))

        top_k = [train_src[x] for x in ranked_examples[0:k]]
        og_src = fxn_src[i].split("__User")[-1]
        og_src = re.sub("__StartOfProgram", "", og_src)
        print(f"original: {og_src}")
        for tk in top_k:
            tk = tk.split("__User")[-1]
            tk = re.sub("__StartOfProgram", "", tk)
            print(f"\tnew: {tk}")


    #print(ranks)
    average_rank = np.mean(ranks)

    print(f"Average rank: {average_rank*100:.2f}")

def manual_eval(path, k=3): 

    with open(path.joinpath("train.src_tok")) as src_f, open(path.joinpath("train.idx")) as idx_f, open(path.joinpath("train.tgt")) as tgt_f:
        train_src = [line.strip() for line in src_f.readlines()]
        train_idxs = [int(line.strip()) for line in idx_f.readlines()]
        train_tgt = [line.strip() for line in tgt_f.readlines()]

    with open(path.joinpath("generated_lookup_table.json")) as f1:
        gen_table = json.load(f1)

    for i, idx in enumerate(gen_table.keys()): 
        ranked_examples = gen_table[idx]
        src_text = train_src[int(idx)]
        tgt_text = train_tgt[int(idx)]
        top_k_src = [train_src[x] for x in ranked_examples[0:k]]
        top_k_tgt = [train_tgt[x] for x in ranked_examples[0:k]]
        og_src = src_text.split("__User")[-1]
        og_src = re.sub("__StartOfProgram", "", og_src)
        print(f"original: {og_src}\n{tgt_text}")
        for tks, tkt in zip(top_k_src, top_k_tgt):
            tks = tks.split("__User")[-1]
            tks = re.sub("__StartOfProgram", "", tks)
            print(f"\tnew: {tks}\n\t{tkt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to conatenated manually annoated min pairs file", required=True)
    parser.add_argument("--k", type=int, default=3, help = "top k outputs to show")
    parser.add_argument("--manual", action="store_true", help = "do manual eval by printing top k instead of computing rank")
    args = parser.parse_args()
    path = pathlib.Path(args.path)
    if not args.manual:
        get_avg_rank(path, args.k)
    else:
        manual_eval(path, args.k)

