import json 
import numpy as np 
from tqdm import tqdm 
from pathlib import Path
import pdb 
from collections import defaultdict, Counter
import scipy
from scipy import stats
import argparse

def get_prediction(prob_dict_list):
    predicted_toks_and_probs = []
    for timestep, prob_dict in enumerate(prob_dict_list): 
        toks, probs = zip(*prob_dict.items())
        # print(prob_dict['SourceCopy'])
        # best_prob_idx = np.argmax(probs)
        top_k_idxs = np.argpartition(probs, -4)[-4:]
        # best_prob, best_tok = probs[top_k_idxs[0]], toks[top_k_idxs[0]]
        top_probs, top_toks = [probs[x] for x in top_k_idxs], [toks[x] for x in top_k_idxs]
        best_prob_idx = np.argmax(top_probs)
        best_tok, best_prob = top_toks[best_prob_idx], top_probs[best_prob_idx]
        predicted_toks_and_probs.append((best_tok, best_prob, top_probs, top_toks))
    return predicted_toks_and_probs

def check_tokens(pred_tok, tgt_tok, prev_tgts):
    if "SourceCopy" not in pred_tok and "TargetCopy" not in pred_tok:
        return pred_tok == tgt_tok
    elif "SourceCopy" in pred_tok:
        return pred_tok.split("_")[1] == tgt_tok
    else:
        try:
            tok_idx = int(pred_tok.split("_")[1])-1
            return prev_tgts[tok_idx] == tgt_tok
        except IndexError:
            print(len(prev_tgts))
            print(pred_tok)
            print(prev_tgts)
            raise AssertionError
    

def read_json(path): 
    print(f"opening data")
    with open(path) as f1:
        data = json.load(f1)
    print(f"got data")
    return data

def get_probs(data):
    probs_to_ret = defaultdict(list)
    func_ontology = set()

    mistakes, corrects = [], []

    for instance_idx, instance in tqdm(enumerate(data)): 
        instance = instance
        left_context = [x[0] for x in instance['left_context']][1:]
        target_toks = left_context + ["@end@"]
        probs = instance['prob_dist']
        predicted_toks = get_prediction(probs)

        source_tokens = " ".join([x[0] for x in instance['source_tokens']])
        for i in range(len(left_context)):
            input_token = left_context[i]
            output_token = predicted_toks[i][0]
            output_prob = predicted_toks[i][1]
            top_k_tokens = predicted_toks[i][2]
            top_k_probs = predicted_toks[i][3]
            target_token = target_toks[i]
            tokens_are_equal = check_tokens(output_token, target_token, left_context[:i])
            if not tokens_are_equal:
                mistake = {"instance_idx": instance_idx,
                           "source_tokens": source_tokens,
                           "is_correct": False,
                           "left_context": left_context[0:i],
                           "target_toks": target_toks,
                           "output_token": output_token,
                           "output_prob": output_prob,
                           "top_k_tokens": top_k_tokens,
                           "top_k_probs": top_k_probs,
                           "target_token": target_token}
                mistakes.append(mistake)
            else:
                correct = {"instance_idx": instance_idx,
                           "source_tokens": source_tokens,
                           "is_correct": True,
                           "left_context": left_context[0:i],
                           "target_toks": target_toks,
                           "output_token": output_token,
                           "output_prob": output_prob,
                           "top_k_tokens": top_k_tokens,
                           "top_k_probs": top_k_probs,
                           "target_token": target_token}
                corrects.append(correct)

    return mistakes, corrects

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logit_file", type=str, default = "/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data//translate_output/test_all_losses.json")
    parser.add_argument("--source_file", type=str, default="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/test_all.src_tok")
    parser.add_argument("--target_file", type=str, default="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/test_all.tgt")
    parser.add_argument("--output_file", type=str, default="hit/data/data_by_bin.json")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--max_per_bin", type=int, default=100) 
    parser.add_argument("--n_bins", type=int, default=20)
    args = parser.parse_args()
    data = read_json(args.logit_file)
    mistakes, corrects = get_probs(data)

    # get all inputs for examples with low confidence, below threshold
    # use stratified sampling across bins to get 100 examples from each bin

    #examples = mistakes + corrects
    # for now, just look at things that had mistakes in them, since we're interested in correcting mistakes  
    examples = mistakes 
    max_n_examples = 100
    example_idxs = [x['instance_idx'] for x in examples]
    # get counter of mistakes per example 
    example_counter = Counter(example_idxs)
    # bin by model confidence 
    values, bins, bin_number = scipy.stats.binned_statistic([x['output_prob'] for x in examples], 
                                                            [x['output_prob'] for x in examples], 
                                                            statistic='mean', 
                                                            bins=args.n_bins)

    idxs_by_bin = {}
    for i in range(1, len(bins)):
        # only look at bins below threshold 
        if bins[i] > args.threshold:
            break
        # get all examples in this bin
        bin_examples = [x for j, x in enumerate(examples) if bin_number[j] == i and example_counter[x['instance_idx']] == 1]
        # skip any bin with no examples in it 
        if len(bin_examples) == 0:
            continue
        # sample some examples from this bin 
        n_examples_per_bin = min(max_n_examples, len(bin_examples))
        sampled_examples = np.random.choice(bin_examples, size=n_examples_per_bin)
        # get the instance idxs 
        instance_idxs = [example['instance_idx'] for example in sampled_examples]
        idxs_by_bin[bins[i]] = instance_idxs


    with open(args.source_file) as f1, \
        open(args.target_file) as f2:
        src_lines = f1.readlines()
        tgt_lines = f2.readlines()

    data_to_write = []
    for bin, idxs in idxs_by_bin.items():
        for idx in idxs:
            try:
                bin_data = {"index": idx, "bin": bin, "source": src_lines[idx], "target": tgt_lines[idx]}
            except IndexError:
                pdb.set_trace()
            data_to_write.append(bin_data)

    with open(args.output_file, "w") as f:
        for line in data_to_write:
            f.write(json.dumps(line) + "\n")