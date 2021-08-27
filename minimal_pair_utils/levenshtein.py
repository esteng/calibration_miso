import string
import pdb 

import numpy as np 
import Levenshtein as lev

STOP = ['in', 'with', 'the', 'and', 'his', 'her', 'my', 'their']
CONJ = ['and', 'or']
PUNC = [',','.','?',':',';']

def charify(s1, s2):
    """
    Levenshtein lib operates over strings of characters, rather than lists of strings
    It's really fast so we want to use it. This is a slightly hacky fix to do that.
    Given 2 lists of strings, get a joint vocab over them, then map each item in the vocab
    to a unique char. Replace the strings with chars, then concatenate. 
    """
    total_vocab = set(s1) | set(s2)
    chars = string.ascii_letters + string.digits + string.punctuation
    try:
        assert(len(total_vocab) < len(chars))
    except AssertionError:
        print("Warning: mapping incomplete, returning large distance to ignore in min")
        return np.inf
    chars = chars[0:len(total_vocab)]
    total_vocab = list(total_vocab)
    mapping = {k:c for k, c in zip(total_vocab, chars)}

    s1 = [mapping[x] for x in s1]
    s2 = [mapping[x] for x in s2]
    s1 = "".join(s1)
    s2 = "".join(s2)
    return (s1, s2, mapping)

def get_swaps(s1, s2): 
    s1, s2, mapping = charify(s1, s2)
    reverse_mapping = {c:k for k,c in mapping.items()}
    op_codes = lev.opcodes(s1, s2)
    swaps = []
    for code in op_codes:
        if code[0] == 'replace':
            __, src_start, src_end, tgt_start, tgt_end = code
            src_str = s1[src_start:src_end]
            tgt_str = s2[tgt_start: tgt_end]
            # for now, skip anything trying to swap multiple words
            if len(src_str) > 3 and len(tgt_str) > 3:
                continue
            src_str = [reverse_mapping[c] for c in src_str]
            tgt_str = [reverse_mapping[c] for c in tgt_str]
            # don't swap punctuation or stopwords 
            if any([x in PUNC for x in src_str]) or any([x in STOP for x in src_str]) or \
             any([x in PUNC for x in tgt_str]) or any([x in STOP for x in tgt_str]):
            
                continue
            # don't swap for the same thing 
            if src_str == tgt_str:
                continue 
            swaps.append((src_str, tgt_str))
    return swaps 

def levenshtein(s1, s2): 
    """Compute Levenshtein distance"""
    # need to turn s1 and s2 vocabs into characters 
    try:
        s1, s2, __ = charify(s1, s2)
    except TypeError:
        pdb.set_trace() 
    return lev.distance(s1, s2)