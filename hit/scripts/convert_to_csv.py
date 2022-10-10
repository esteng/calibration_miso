import argparse
import csv 
import json 
import pathlib 
import numpy as np 

from prep_for_translate import split_source

def read_json(input_file):
    with open(input_file) as f1:
        data = [json.loads(x) for x in f1.readlines()]
    # remove duplicates (when there are multiple low-confidence decisions)
    data_by_unique_id = {x['data_idx']: x for x in data}
    return list(data_by_unique_id.values())

def convert_data_to_csv(all_data, out_dir): 
    csv_data = []
    for line in all_data: 
        # need user turns 
        turns = split_source(line['gold_src'])
        if len(turns) == 1: 
            # no previous turns 
            turns = ["", "", turns[0]]
        line_data = {"user_turn_0": turns[0],
                    "agent_turn_0": turns[1],
                    "user_turn_1": turns[2]} 
        options = line['pred_translated'] + [line['distractor']]
        options_and_idxs = [(x, i) for i, x in enumerate(options)]
        np.random.shuffle(options_and_idxs)
        options, idxs = zip(*options_and_idxs)
        for i in range(len(options)): 
            line_data[f"option_{i}"] = options[i]
            line_data[f"option_{i}_idx"] = idxs[i]
        
        csv_data.append(line_data) 
    
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / "data.csv", 'w') as f1: 
        writer = csv.DictWriter(f1, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)
    with open(out_dir / "data.json", 'w') as f1: 
        for line in all_data:
            f1.write(json.dumps(line) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    json_data = read_json(args.input)
    convert_data_to_csv(json_data, args.out_dir)