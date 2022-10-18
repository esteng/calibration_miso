import argparse
import csv 
import json 
import pdb 
import pathlib 
import numpy as np 
np.random.seed(12)

from prep_for_translate import split_source

def read_json(input_file, ignore_idxs=[]):
    with open(input_file) as f1:
        data = [json.loads(x) for x in f1.readlines()]
    # remove duplicates (when there are multiple low-confidence decisions)
    
    data_by_unique_id = {x['data_idx']: x for x in data if x['data_idx'] not in ignore_idxs}
    return list(data_by_unique_id.values())

def convert_data_to_csv(all_data, 
                        out_dir, 
                        shuffle=False, 
                        no_option_shuffle=False,
                        no_distractor=False): 
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

        if no_distractor:
            options = line['pred_translated'] 
            min_probs = line['min_probs'] 
        else:
            options = line['pred_translated'] + [line['distractor']]
            min_probs = line['min_probs'] + [0.0]

        zipped_options_and_probs = zip(options, min_probs)
        options_and_idxs = [(x, i, p) for i, (x, p) in enumerate(zipped_options_and_probs)]

        # if flag set to not shuffle options, keep sorted by best total prob 
        if no_option_shuffle:
            pass
        else:
            np.random.shuffle(options_and_idxs)

        options, idxs, min_probs = zip(*options_and_idxs)
        for i in range(len(options)): 
            line_data[f"option_{i}"] = options[i]
            line_data[f"option_{i}_idx"] = idxs[i]
            line_data[f"prob_{i}"] = min_probs[i]
        
        csv_data.append(line_data) 
    
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if shuffle:
        zipped_data = list(zip(csv_data, all_data))
        np.random.shuffle(zipped_data)
        csv_data, all_data = zip(*zipped_data)

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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ignore_idx_file", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no_option_shuffle", action="store_true")
    parser.add_argument("--no_distractor", action="store_true")
    args = parser.parse_args()

    # add ability to ignore indices if they've already been included in previous hits 
    if args.ignore_idx_file is not None:
        with open(args.ignore_idx_file) as f1:
            ignore_idxs = [json.loads(x)['data_idx'] for x in f1.readlines()]
    else:
        ignore_idxs = []
    json_data = read_json(args.input, ignore_idxs)

    # adding ability to write only a subset of data 
    if args.limit is not None:
        json_data = json_data[:args.limit]

    convert_data_to_csv(json_data, 
                        args.out_dir, 
                        args.shuffle, 
                        args.no_option_shuffle,
                        args.no_distractor)