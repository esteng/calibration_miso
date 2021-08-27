import pdb 
import logging 
import re
import json 

import Levenshtein as lev
import numpy as np
from levenshtein import levenshtein, get_swaps, PUNC, STOP, CONJ 


class Mutation:
    def __init__(self):
        pass 

class DeleteMutation(Mutation): 
    def __init__(self): 
        pass 

    def __call__(self, source):
        # base case, nothing to delete 
        if len(source) == 1:
            return source
        # choose a span to delete 
        span_idxs = [i for i in range(1, len(source))]
        span_start = np.random.choice(span_idxs[:-1])
        span_end = np.random.choice(span_idxs[span_idxs.index(span_start) + 1:])
        # apply heuristics 
        span_start, span_end = self.apply_heuristics(span_start, span_end, source)
        #print(f"excising {source[span_start:span_end]} from {span_start} to {span_end}")
        new_source = source[0:span_start] + source[span_end:]
        #print(f"DELETE: source: {' '.join(source)}\n\tnew source: {' '.join(new_source)}")
        return new_source

    def apply_heuristics(self, start, end, text): 
        # heuristics: if you delete only stopword, delete the next word as well
        if text[start] in STOP and end == start + 1:
            end = start + 2
            #return start, end
        # if you delete a word where the previous word is a stopword, delete the stopword 
        if start > 0 and text[start-1] in STOP: 
            start -= 1
            #return start, end 
        # if you delete a word where the previous word is a conjunction, delete the stopword and everything after
        if start > 0 and text[start-1] in CONJ:
            start -= 1
            end = len(text)
            #return start, end 
        # if you delete a conjunction, delete everything after the conjunction 
        text_to_cut = text[start:end]
        for conj in CONJ:
            if conj in text_to_cut:
                end = len(text)
                break
                #return start, end 
        return start, end

class SwapMutation(Mutation):
    def __init__(self, source, nearby_train):
        super().__init__()

        swaps = []
        for target in nearby_train:
            swaps += get_swaps(source, target)
        self.swaps = swaps 

    def __call__(self, source):
        valid_swap = self.swaps[np.random.choice(len(self.swaps))]
        new_source = [x for x in source]
        swap_len = len(valid_swap[0])
        swap_src_span = " ".join(valid_swap[0])
        for i in range(len(source)-swap_len + 1):
            #print(valid_swap[0][0], word)
            span = " ".join(source[i:i+swap_len])
            if span == swap_src_span:
                #print(f"swapping {span} for {' '.join(valid_swap[1])}")
                new_source[i:i+swap_len] = valid_swap[1]

        #print(f"SWAP: source: {' '.join(source)}\n\tnew source: {' '.join(new_source)}")
        return new_source 

    @classmethod
    def from_train(cls, source, train_lines, k=120):
        dists = np.ones(len(train_lines)) * np.inf
        for i, line in enumerate(train_lines):
            dists[i] = levenshtein(source, line)
        top_k_idxs = np.argpartition(dists, k)[:k]
        nearby_train = [train_lines[i] for i in top_k_idxs]

        return cls(source, nearby_train)

        

     

class AnonSwapMutation(SwapMutation):
    def __init__(self, source, nearby_train, names):
        super().__init__(source, nearby_train)
        source = anonymize_input(source, names)
        self.names = names 

    def __call__(self, source):
        source = anonymize_input(source, self.names)
        valid_swap = self.swaps[np.random.choice(len(self.swaps))]
        new_source = [x for x in source]
        swap_len = len(valid_swap[0])
        swap_src_span = " ".join(valid_swap[0])
        for i in range(len(source)-swap_len + 1):
            #print(valid_swap[0][0], word)
            span = " ".join(source[i:i+swap_len])
            if span == swap_src_span:
                #print(f"swapping {span} for {' '.join(valid_swap[1])}")
                new_source[i:i+swap_len] = valid_swap[1]
        return new_source 

    @classmethod
    def from_train(cls, source, train_lines, names, k=120):
        source = anonymize_input(source, names)
        dists = np.ones(len(train_lines)) * np.inf
        for i, src_line in enumerate(train_lines):
            src_line = anonymize_input(src_line, names)
            #print(src_line)
            dists[i] = levenshtein(source, src_line)
        top_k_idxs = np.argpartition(dists, k)[:k]
        nearby_train = [train_lines[i] for i in top_k_idxs]
        print(source)
        print(nearby_train)
        #print("\n".join([" ".join(x) for x in nearby_train]))
        #pdb.set_trace() 
        return cls(source, nearby_train, names)
     
def anonymize_input(source_line, names): 
    # want to substitute names in the input with NAME 
    for span_len in range(len(source_line)-1, 0, -1):
        for span_start in range(0, len(source_line)-span_len):
            span_end = span_start + span_len
            span = source_line[span_start:span_end]
            if " ".join(span) in names:
                for i in range(span_start, span_end):
                    source_line[i] = "NAME"
    return source_line 

class IdentityMutation(Mutation):
    def __init__(self):
        super().__init__()
    
    def __call__(self, source): 
        return source