import argparse
from pathlib import Path 
from collections import defaultdict
import json 
import pdb 

from tqdm import tqdm 
import numpy as np 

from dataflow.core.linearize import lispress_to_seq, seq_to_lispress
from dataflow.core.lispress import parse_lispress, program_to_lispress, lispress_to_program
from dataflow.core.program import ValueOp, Expression

from mutations import DeleteMutation, IdentityMutation, SwapMutation, AnonSwapMutation, anonymize_input
from levenshtein import levenshtein

def read_data(path, fxn=False): 
    path = Path(path) 
    if fxn:
        prefix = "fxn_"
    else:
        prefix = ""
    src_path = path.joinpath(f"{prefix}train.src_tok") 
    idx_path = path.joinpath(f"{prefix}train.idx") 
    tgt_path = path.joinpath(f"{prefix}train.tgt") 

    with open(src_path) as f1:
        src_lines = [line.strip().split(" ") for line in f1.readlines()]
        new_src_lines = []
        for src_line in src_lines:
            user_idxs = [i for i, x in enumerate(src_line) if x == "__User"]
            new_src_line = src_line[user_idxs[-1] + 1: -1]
            new_src_lines.append(new_src_line)
    with open(tgt_path) as f1:
        tgt_lines = [line.strip().split(" ") for line in f1.readlines()]
    with open(idx_path) as f1:
        idx_lines = [int(line.strip()) for line in f1.readlines()]
    return new_src_lines, idx_lines, tgt_lines

def choose_fxn(fxn_tgt, fxn_frequencies): 
    fxns_in_tgt = [x for x in fxn_tgt if x in fxn_frequencies.keys()]
    text_freq = {k:0 for k in set(fxns_in_tgt)}
    for fxn in fxns_in_tgt:
        text_freq[fxn] += 1

    tfidfs = {text_freq[fxn] / fxn_frequencies[fxn] for fxn in set(fxns_in_tgt)}

    pdb.set_trace() 

def anonymize_tgt(lispress_seq):
    lispress = seq_to_lispress(lispress_seq)
    program, __ = lispress_to_program(lispress, 0)

    for i, expr in enumerate(program.expressions):
        op = expr.op
        if isinstance(op, ValueOp):
            value = json.loads(op.value)
            if value['schema'] == "String":
                value['underlying'] = "String"
            elif value['schema'] == "Long": 
                value['underlying'] = 0
            elif value['schema'] == "Boolean":
                value['underlying'] = True
            elif value['schema'] == "Number":
                value['underlying'] = 0
            else:
                pdb.set_trace()
            new_op = ValueOp(json.dumps(value))
            new_expr = Expression(expr.id, new_op, expr.type_args, expr.type, expr.arg_ids)
            program.expressions[i] = new_expr 
    
    lispress = program_to_lispress(program)
    seq = lispress_to_seq(lispress)
    return seq

def share_words(s1, s2):
    return any([x in s2 for x in s1])


def restrict_mutants(mutants, 
                    fxn_tgt,    
                    train_src, 
                    train_tgt, 
                    train_idxs, 
                    fxn_of_interest=None, 
                    names=None, 
                    do_sum = False, 
                    fxn_frequencies=None, 
                    anon_plan=False,
                    top_k=-1,
                    heuristics=False): 
    """
    This function is key. Enforces target constraints on the mutant, namely that 
    fxn of interest doesn't appear in target (minimal pair).
    Since we're doing k x N comparisons anyway, also compute distance here, ignoring those 
    that are infeasible.
    """
    # remove any mutants whose train tgt contains fxn 
    final_mutants = []
    # set all distances to inf so that infeasible ones never get chosen
    if top_k == -1:
        full_dists = np.ones((len(mutants), len(train_src))) * np.inf   
    else:
        full_dists = np.ones((len(mutants), top_k)) * np.inf   

    counter = -1
    min_dist = np.inf 
    for j, (src, tgt, idx) in enumerate(zip(train_src, train_tgt, train_idxs)): 
        counter +=1 

        if do_sum:
            tgt_dist = levenshtein(fxn_tgt, tgt)

        for i, mutant in enumerate(mutants):
            if top_k == -1:
                tj = j
            else:
                tj = counter 

            # heuristics 
            # if they're the same, skip
            if mutant == src: 
                continue
            # if the lengths are so different that it doesn't possibly have a 
            # chance of being the min, skip 
            if abs(len(mutant) - len(src)) >= min_dist:
                continue
            # if they're above a length of 1 and share 0 words, skip
            if not(share_words(mutant, src)) and len(mutant) > 1:
                continue

            if not anon_plan:
                if fxn_of_interest is None:
                    fxn_of_interest = choose_fxn(fxn_tgt, fxn_frequencies)
                if fxn_of_interest in tgt:
                    continue 
            else:
                anon_tgt = anonymize_tgt(tgt)
                anon_fxn_tgt = anonymize_tgt(fxn_tgt)
                if anon_tgt == anon_fxn_tgt:
                    continue 

            if names is not None:
                src = anonymize_input(src, names)
            src_dist =  levenshtein(mutant, src) 

            if src_dist < min_dist:
                min_dist = src_dist 

            if do_sum:
                full_dists[i,tj] = src_dist + tgt_dist 
            else:
                full_dists[i,tj] = src_dist 
            final_mutants.append(mutant)
    return final_mutants, full_dists

def get_best_mutants(train_src, 
                     train_idxs, 
                     train_tgt, 
                     fxn_src, 
                     fxn_tgt, 
                     fxn_of_interest, 
                     num_mutants = 10, 
                     names=None, 
                     mutation_types=["delete", "swap"], 
                     do_sum = False, 
                     fxn_frequencies=None,
                     anon_plan=False,
                     top_k=-1): 
    mutations = []
    if "delete" in mutation_types:
        mutations.append(DeleteMutation())
    if "swap" in mutation_types:
        mutations.append(SwapMutation.from_train(fxn_src, train_src))
    if "anon_swap" in mutation_types:
        assert(names is not None)
        mutations.append(AnonSwapMutation.from_train(fxn_src, train_src, names))
    if "identity" in mutation_types: 
        mutations.append(IdentityMutation())

    mutants = []
    if top_k == -1:
        dist_across_mutants = np.zeros((len(mutations) * num_mutants, len(train_src)))
    else:
        dist_across_mutants = np.zeros((len(mutations) * num_mutants, top_k))

    for i in range(num_mutants):
        sources = [m(fxn_src) for m in mutations]
        final_mutants, full_dist = restrict_mutants(sources, 
                                                    fxn_tgt, 
                                                    train_src, 
                                                    train_tgt, 
                                                    train_idxs, 
                                                    fxn_of_interest, 
                                                    do_sum=do_sum,
                                                    fxn_frequencies=fxn_frequencies, 
                                                    anon_plan = anon_plan,
                                                    top_k = top_k) 

        dist_across_mutants[len(sources)*i:len(sources)*i+len(sources)] = full_dist 

    # choose a mutant for each train datapoint 
    best_mutant_idx_by_train = np.argmin(dist_across_mutants, axis=0)
    best_train_dists = np.take_along_axis(dist_across_mutants, best_mutant_idx_by_train.reshape(1, -1), axis=0)
    return best_train_dists

def sort_train_by_min_pair(train_src, 
                           train_idxs, 
                           train_tgt, 
                           fxn_src, 
                           fxn_tgt, 
                           fxn_of_interest, 
                           num_mutants = 10, 
                           names=None, 
                           mutation_types=["delete", "swap"], 
                           do_sum=False,
                           fxn_frequencies=None,
                           anon_plan=False,
                           top_k=-1): 
    best_train_dists = get_best_mutants(train_src, 
                                        train_idxs, 
                                        train_tgt, 
                                        fxn_src, 
                                        fxn_tgt, 
                                        fxn_of_interest, 
                                        num_mutants, 
                                        names, 
                                        mutation_types, 
                                        do_sum=do_sum, 
                                        fxn_frequencies=fxn_frequencies,
                                        anon_plan=anon_plan,
                                        top_k=top_k)
    # sort idxs 
    best_train_idxs = np.argsort(best_train_dists.reshape(-1))
    # reorder train idxs 
    return best_train_idxs.tolist()         

def bucket_by_src(train_src, train_idxs, train_tgt, bucket_size):
    print(f"getting lengths for {len(train_src)} examples")
    train_lengths = [len(x) for x in train_src]
    max_len = np.max(train_lengths) 
    max_len = max_len + max_len % bucket_size
    print(f"Making buckets in increments of size: {bucket_size}")
    buckets_by_len = {i:{"src":[], "idx": [], "tgt":[]} for i in range(0, max_len, bucket_size)} 
    for src, idx, tgt, l in zip(train_src, train_idxs, train_tgt, train_lengths):
        bucket_idx = l - l % bucket_size 
        buckets_by_len[bucket_idx]['src'].append(src)
        buckets_by_len[bucket_idx]['idx'].append(idx)
        buckets_by_len[bucket_idx]['tgt'].append(tgt)
    return buckets_by_len


def main(args):
    train_src, train_idxs, train_tgt = read_data(args.train_path)
    if Path(args.fxn_path).joinpath("fxn_train.src_tok").exists():
        fxn_src, fxn_idxs, fxn_tgt = read_data(args.fxn_path, fxn=True)
    else:
        fxn_src, fxn_idxs, fxn_tgt = train_src, train_idxs, train_tgt

    if args.names is not None:
        names = json.load(open(args.names))
    else:
        names = args.names

    mutation_types = [x.strip() for x in args.mutation_types.split(",")]
    # keys are idxs, values are sorted list of train lines 
    min_pair_lookup = defaultdict(list)

    # TODO: need to so something smart here where we bucket src into lengths, only consider things within a 
    # certain length of the original 
    if args.bucket_size > -1:
        bucketed_data = bucket_by_src(train_src, train_idxs, train_tgt, bucket_size=args.bucket_size)

    for src, idx, tgt in tqdm(zip(fxn_src, fxn_idxs, fxn_tgt)):
        if args.bucket_size > -1:
            src_len = len(src) - len(src) % args.bucket_size
            bucket_subset = bucketed_data[src_len]
            bucket_src = bucket_subset['src']
            bucket_idxs = bucket_subset['idx']
            bucket_tgt = bucket_subset['tgt']
            print(f"bucket: {src_len} has {len(bucket_src)} examples")
        else:
            bucket_src = train_src
            bucket_idxs = train_idxs
            bucket_tgt = train_tgt 

        min_pair_lookup[idx] = sort_train_by_min_pair(bucket_src, 
                                                      bucket_idxs, 
                                                      bucket_tgt, 
                                                      src, 
                                                      tgt, 
                                                      args.fxn_of_interest, 
                                                      num_mutants=args.num_mutants, 
                                                      names = names, 
                                                      mutation_types = mutation_types, 
                                                      do_sum = args.do_sum,
                                                      fxn_frequencies=None,
                                                      anon_plan=args.anon_plan,
                                                      top_k=args.top_k)

    with open(args.out_path, "w") as f1:
        json.dump(min_pair_lookup, f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, help="path to dir containing train files (src_tok, idx, tgt)", required = True)
    parser.add_argument("--fxn-path", type=str, help = "path to dir containing fxn files (src_tok, idx, tgt)", required=True)
    parser.add_argument("--fxn-of-interest", type=str, help="function to construct minimal pairs for", required=False)
    parser.add_argument("--names", type=str, help = "location of names json", required=False)
    parser.add_argument("--mutation-types", type=str, help="types of mutations to apply. Choices are delete, swap, identity, anon_swap", default="delete,swap")
    parser.add_argument("--out-path", type=str, help = "path to write output")
    parser.add_argument("--do-sum", action="store_true", help = "use sum of source and target distance instead of just source distance")
    parser.add_argument("--fxn-frequencies", type=str, help="path to json of function frequencies if using TFIDF")
    parser.add_argument("--anon-plan", action="store_true", help="instead of doing TFIDF for sampling in full data setting, just ensure that the anonymized plans aren't equal")
    parser.add_argument("--num-mutants", type=int, default=10, help="number of mutants to consider")
    parser.add_argument("--top-k", type=int, default=-1, help = "number of pairs to get per example (-1 means rank the whole train set)")
    parser.add_argument("--bucket-size", type=int, default=-1, help = "size of the bucket for length bucketing")
    args = parser.parse_args()
    main(args)


