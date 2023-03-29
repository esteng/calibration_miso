import json 
import pdb
import re 
import numpy as np 
import argparse 
import pathlib

from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import pad_sequence, padded_everygram_pipeline
from nltk.util import ngrams
from transformers import AutoTokenizer

def read_data(path):
    with open(path, 'r') as f:
        data = [json.loads(x) for x in f.readlines()]
    return data

def read_test_data(path):
    with open(path) as f1:
        data = f1.readlines()
    return [x.strip() for x in data]

def get_and_tokenize(data, key, tokenizer, n = 5, pad=False):
    assert key in ['plan', 'utterance']

    to_ret = []
    pad_len = n

    for d in data:
        item = d[key]
        # tokenize
        split_item = tokenizer.tokenize(item)
        if pad:
            split_item = [x for x in pad_sequence(split_item, pad_len, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')]
        to_ret.append(split_item)

    return to_ret

def train_lm(train_sents, vocab, n = 5):
    model = KneserNeyInterpolated(n)
    model.fit(train_sents, vocab)
    return model

def get_sent_ppl(model, sent):
    ngram_seq = list(ngrams(sent, model.order))
    # pdb.set_trace()
    ppl = model.perplexity(ngram_seq)
    return ppl 

def get_ppl(model, sentences):
    ppls = []
    for sent in sentences:
        ppl = get_sent_ppl(model, sent)
        if np.isinf(ppl):
            continue
        ppls.append(ppl)

    print(f"\tppls: {ppls}, len(ppls): {len(ppls)}")
    print(f"\tfailed to compute ppl for {len(sentences) - len(ppls)} sentences")
    return np.sum(ppls)/len(sentences)

class WhitespaceTokenizer:
    def tokenize(self, text):
        return [x.lower() for x in re.split("\s+", text)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=None, required=True)
    parser.add_argument("--test_dir", type=pathlib.Path, default=None, required=True)
    parser.add_argument("--use_target", action='store_true')
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--order", type=int, default=5, required=False)
    parser.add_argument("--max_len", type=int, default=None)
    args = parser.parse_args()

    train_data = read_data(args.train_file)
    if args.use_target:
        key = "plan"
    else:
        key = "utterance"

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 
    tokenizer = WhitespaceTokenizer()

    train_sents = get_and_tokenize(train_data, key, tokenizer, args.order)
    if args.max_len is not None:
        train_sents = train_sents[0:args.max_len]

    train_data, vocab = padded_everygram_pipeline(args.order, train_sents)
    train_data = [list(x) for x in train_data]
    model = train_lm(train_data, vocab, args.order)
    
    conf_and_ppl = []
    for test_file in args.test_dir.glob("*.jsonl"):
        bin_confidence = float(test_file.stem.split("_")[-1])
        print(f"file : {test_file}, confidence: {bin_confidence}")
        test_data = read_data(test_file)
        test_sents = get_and_tokenize(test_data, key, tokenizer, args.order, pad=True)
        mean_ppl = get_ppl(model, test_sents)
        # mean_ppl = np.mean(ppls)
        conf_and_ppl.append((bin_confidence, mean_ppl))


    conf_and_ppl = sorted(conf_and_ppl, key=lambda x: x[0])
    for conf, ppl in conf_and_ppl:
        print(f"Confidence: {conf}, Average PPL: {ppl}")