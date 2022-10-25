import csv 
import pdb 
import re 
import json 
import argparse 


def split_source(source):
    source = re.sub("__StartOfProgram", "", source)
    source = source.strip()
    split_source = re.split("(__User)|(__Agent)", source)
    split_source = [x for x in split_source if x not in [None, '']]
    last_user = ""
    last_agent = ""
    curr_user = "" 
    if len(split_source) == 2: 
        curr_user = split_source[-1]
    elif len(split_source) == 6:
        last_user = split_source[1]
        last_agent = split_source[3]
        curr_user = split_source[5]
    return last_user, last_agent, curr_user

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    csv_lines = []    
    with open(args.jsonl_file) as f1:
        for line in f1.readlines():
            data = json.loads(line)
            user_utt_0, agent_utt_0, user_utt_1 = split_source(data['gold_src'])
            __, __, paraphrase = split_source(data['pred_src'])

            csv_line = {"user_turn_0": user_utt_0, 
                        "agent_turn_0": agent_utt_0, 
                        "user_turn_1": user_utt_1, 
                        "paraphrase": paraphrase,
                        "confidence": f"{data['confidence']*100:.0f}%"}
            csv_lines.append(csv_line)


    with open(args.out_file, "w") as f1: 
        writer = csv.DictWriter(f1, fieldnames=csv_lines[0].keys())
        writer.writeheader()
        for line in csv_lines:
            writer.writerow(line)