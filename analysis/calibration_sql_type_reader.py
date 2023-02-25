import json 
from collections import defaultdict
import re 
import pdb 
from typing import Tuple, List, Any
import numpy as np

from transformers import AutoTokenizer

from calibration_metric.utils.reader import Reader

class TypeTopLogitFormatSequenceReader(Reader):
    """
    Dataset reader for the HF output format.
    File format is jsonl, where each line is a 
    dict corresponding to a single input line,
    with the following keys:
    - top_logits: list of the top k logits for each timestep
    - top_logit_idxs: list of the top k logits for each timestep
    - labels: list of the label indices for each timestep
    """
    def __init__(self, file: str, ignore_tokens: List[Any] = None, model_name: str = "t5-small"):
        self.file = file
        self.ignore_tokens = ignore_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def parse_line_into_types(self, top_logits: np.array, labels: List):
        sql_funcs = ["SELECT", "WHERE", "GROUP BY", "HAVING", "ORDER", "BY", "FROM",
                    "LIMIT", "JOIN", "INTERSECT", "EXCEPT", "UNION", "DESC", "ASC",
                    "NOT IN", "OR", "AND", "EXISTS", "LIKE", "DISTINCT", "GROUP",
                    "BETWEEN", "AS", "ON", "NOT", "IN"]
        # get strs 
        labels_as_toks = self.tokenizer.convert_ids_to_tokens(labels)
        # convert to detokenized 
        tok_idx_to_str_idx = {}
        str_idx_to_tok_idx = defaultdict(list)
        
        missing = []
        str_toks = []
        curr_tok = []
        str_idx = -1
        # no eos 
        for i, tok in enumerate(labels_as_toks[0:-1]):
            # is not a subword 
            if tok.startswith("▁"):
                if len(curr_tok) > 0:
                    str_toks.append(curr_tok)
                # start of a new token  
                curr_tok = [tok]
                str_idx += 1
                tok_idx_to_str_idx[i] = str_idx
                str_idx_to_tok_idx[str_idx].append(i)

            # is a subword 
            else:
                # add to curr tok
                curr_tok.append(tok)
                tok_idx_to_str_idx[i] = str_idx
                str_idx_to_tok_idx[str_idx].append(i) 

        # add last token
        str_toks.append(curr_tok)
        # parse line into types 
        # 1 = function, 2 = column, 3 = value, 4 = other
        types = np.zeros_like(labels)
        prev_was_fxn = False
        prev_was_paren = False
        prev_was_quote = False

        for i, token in enumerate(str_toks): 
            token = "".join(token)
            token = re.sub("^▁", "", token)
            # pdb.set_trace()
            type_idxs = str_idx_to_tok_idx[i]
            # rules:
            if token in sql_funcs:  
                for type_idx in type_idxs:
                    types[type_idx] = 1
                prev_was_fxn = True
                prev_was_paren = False
                prev_was_quote = False
                continue
            # if the previous token is a function then this is a column 
            elif prev_was_fxn:
                for type_idx in type_idxs:
                    types[type_idx] = 2
                prev_was_fxn = False
                prev_was_paren = False
                prev_was_quote = False
            # if the token is surrounded by () then it is a column 
            elif token in ['(', '.', ',']:
                prev_was_fxn = False
                prev_was_paren = True
                prev_was_quote = False
            elif prev_was_paren or re.match("\(.+\)", token) or re.match("T\d+", token):
                for type_idx in type_idxs:
                    types[type_idx] = 2
                # prev_was_fxn = False
                prev_was_paren = False
                prev_was_quote = False

            # if the token is surrounded by "" then it is a value
            elif token in ['"', "'"]:
                prev_was_fxn = False
                prev_was_paren = False
                prev_was_quote = True
            elif prev_was_quote or token.startswith('"') or token.startswith("'") or re.match("\d+", token):
                for type_idx in type_idxs:
                    types[type_idx] = 3
                prev_was_fxn = False
                prev_was_paren = False
                prev_was_quote = False
            # check if last token was a value, could be double value  
            elif i > 0 and types[str_idx_to_tok_idx[i-1][-1]] == 3:
                for type_idx in type_idxs:
                    types[type_idx] = 3
                prev_was_fxn = False
                prev_was_paren = False
                prev_was_quote = False
            else:
                missing.append(token)
        zipped = list(zip(labels_as_toks, types))

        allowed_missing = [')', "=", ">", "<", ">=", "<=", "<extra_id_99>", ";", "<extra_id_99>=", "!="]
        for m in missing:
            if m not in allowed_missing:
                pass 
                # pdb.set_trace()
        return types 

    def read(self) -> Tuple[np.array]:
        """
        Read the file and extract the single top predicted index
        and the corresponding confidence score (logit).
        Compares each predicted index to the label index to determine
        whether the prediction was correct.
        Returns:
            top_preds: np.array of shape (num_examples, )
            is_correct: np.array of shape (num_examples, )
        """
        all_top_preds = [[] for i in [0, 1, 2, 3]]
        all_is_correct = [[] for i in [0, 1, 2, 3]]

        with open(self.file, 'r') as f:
            for line in f:
                line = json.loads(line)
                top_k_logits = np.array(line['top_logits'])
                top_k_logit_idxs = np.array(line['top_logit_idxs'])
                # get the top 1 logit and idx
                top_one_logit_local_idx = np.argmax(top_k_logits, axis=-1)
                seq_len = top_one_logit_local_idx.shape

                if len(seq_len) > 1: 
                    # TODO: implement batched version
                    raise NotImplementedError(f"Currently batched outputs are not supported.\
                         Try generating outputs a single example at a time.")

                seq_len = seq_len[0]

                # get the actual single top logit, not assuming they're sorted already 
                top_one_logit_local_idx = top_one_logit_local_idx.reshape((seq_len, 1))
                top_one_logit = np.take_along_axis(top_k_logits, top_one_logit_local_idx, axis=1)
                top_one_logit_idx = np.take_along_axis(top_k_logit_idxs, top_one_logit_local_idx, axis=1)
                labels = np.array(line['labels'])


                # currently only support single example per line 
                top_logits = top_one_logit.reshape(-1)
                top_logit_idxs = top_one_logit_idx.reshape(-1)

                types = self.parse_line_into_types(top_logit_idxs, labels)

                for type_type in [0, 1, 2, 3]:
                    top_logits_for_type = top_logits[types == type_type]
                    labels_for_type = labels[types == type_type]

                    is_correct = self.check_tokens(top_logits_for_type, labels_for_type) 

                    for timestep in range(top_logits_for_type.shape[0]):
                        # ignore tokens if specified
                        # meant to be tokens like @ROOT@, BOS, EOS, etc.
                        if self.ignore_tokens is not None and labels_for_type[timestep] in self.ignore_tokens:
                            continue
                        all_top_preds[type_type].append(top_logits_for_type[timestep])
                        all_is_correct[type_type].append(is_correct[timestep])

        for i in [0, 1, 2, 3]:
            all_top_preds[i] = np.array(all_top_preds[i])
            all_is_correct[i] = np.array(all_is_correct[i])
        return all_top_preds, all_is_correct
        # return (np.array(all_top_preds), np.array(all_is_correct))

if __name__ == "__main__":
    path = "/brtx/604-nvme2/estengel/calflow_calibration/benchclamp/1.0/t5-small-lm-adapt_spider_past_none_db_val_all_0.0001/checkpoint-10000/outputs/test_all.logits"

    reader = TypeTopLogitFormatSequenceReader(path)
    top_preds, is_corect = reader.read()
    pdb.set_trace()