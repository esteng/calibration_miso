import json 
import re 
from collections import defaultdict
import argparse

import spacy 
import sys

from dataflow.core.utterance_tokenizer import UtteranceTokenizer
from dataflow.core.lispress import parse_lispress
from dataflow.core.linearize import lispress_to_seq

tokenizer = UtteranceTokenizer()
def tokenize(test):
    return " ".join(tokenizer.tokenize(test))

def main(args):
    to_write = []
    with open(args.text_pred_file) as predf, open(args.data_jsonl) as jsonf:
        texts = [line for line in predf]
        jsons = [json.loads(line) for line in jsonf]
        for text, jsonl in zip(texts, jsons):
            text = re.sub("<.*?>", "", text)
            text = re.sub(" \| ", "", text)
            text = text.strip() 
            if not args.use_gold:
                jsonl["utterance"] = text
            to_write.append(json.dumps(jsonl))
    if args.out_format == "jsonl":
        with open(args.out_file, "w") as f:
            for line in to_write:
                f.write(line + "\n") 

    elif args.out_format == "lines": 
        out_src = args.out_file + ".src_tok"
        out_tgt = args.out_file + ".tgt"
        with open(out_src, "w") as srcf, open(out_tgt, "w") as tgtf:
            for line in to_write: 
                line = json.loads(line)
                # 
                user_utt = tokenize(line['utterance'])
                agent_utt = tokenize(line['last_agent_utterance'])
                last_user_utt = tokenize(line['last_user_utterance'])
                text = []
                if len(last_user_utt) > 0: 
                    text.append("__User")
                    text.append(last_user_utt)
                if len(agent_utt) > 0:                
                    text.append("__Agent")
                    text.append(agent_utt)
                text.append("__User")
                text.append(user_utt)
                text.append("__StartOfProgram")
                text = " ".join(text).strip() 

                srcf.write(text + "\n") 
                lispress = " ".join(lispress_to_seq(parse_lispress(line['plan']))) 
                tgtf.write(lispress + "\n")


    else: 
        raise ValueError(f"Unknown out_format {args.out_format}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_pred_file", type=str, default=None, required=True) 
    parser.add_argument("--data_jsonl", type=str, default=None, required=True)
    parser.add_argument("--out_file", type=str, default=None, required=True)
    parser.add_argument("--out_format", type=str, default="jsonl", required=False)
    parser.add_argument("--use_gold", action="store_true", default=False)
    args = parser.parse_args() 
    main(args)