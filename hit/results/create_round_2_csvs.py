import csv 
import json 
from collections import defaultdict
import pathlib 
import argparse
import pdb 

def clean_str(str): 
    if '"' in str:
        str = str.replace('"', '\\"')
    return str

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    # read in csv
    with open(args.input_csv, "r") as f:
        reader = csv.DictReader(f)
        input_data = list(reader)

    # read in results csv
    with open(args.results_csv, "r") as f:
        reader = csv.DictReader(f)
        results_data = list(reader)

    # match up results with input data
    assert(len(input_data) == len(results_data))

    # for each annotator, get the lines they did
    annotators = [res_line['WorkerId'] for res_line in results_data]
    annotators = list(set(annotators))

    res_keys_by_anns = defaultdict(list)
    for ann in annotators:
        res_lines_for_ann = [x for x in results_data if x['WorkerId'] == ann]
        res_keys_for_ann = [(x["Input.user_turn_0"].strip(), x["Input.agent_turn_0"].strip(), x["Input.user_turn_1"].strip()) for x in res_lines_for_ann]
        res_keys_by_anns[ann] = res_keys_for_ann
        # input_for_ann = [x for x in input_data if (x["user_turn_0"], x["agent_turn_0"], x["user_turn_1"]) not in res_keys_for_ann]]            
        # data_by_anns[ann] = input_for_ann

    # for each annotator, get the lines they did and assign them evenly to other annotators 
    lines_for_anns = {ann: [] for ann in annotators}
    for ann in annotators:
        keys_done_by_ann = res_keys_by_anns[ann]
        lines_done_by_ann = [x for x in input_data if (x["user_turn_0"].strip(), x["agent_turn_0"].strip(), x["user_turn_1"].strip()) in keys_done_by_ann]
        print(f"{ann} did {len(lines_done_by_ann)} lines")
        other_anns = set(annotators) - set([ann])
        for i, other_ann in enumerate(other_anns):
            start = i * len(lines_done_by_ann) // len(other_anns)
            end = (i + 1) * len(lines_done_by_ann) // len(other_anns)
            lines_for_anns[other_ann] += lines_done_by_ann[start:end]

    assert(sum([len(x) for x in lines_for_anns.values()]) == len(input_data))

    # write out the lines for each annotator, adding key for is_checkmark_page
    for ann in annotators:
        file = out_dir / f"{ann}.csv"
        with open(file, "w") as f:
            to_write = []
            for i, line in enumerate(lines_for_anns[ann]):
                line['user_turn_0'] = clean_str(line['user_turn_0'])
                line['agent_turn_0'] = clean_str(line['agent_turn_0'])
                line['user_turn_1'] = clean_str(line['user_turn_1'])
                if i == 0:
                    line['is_checkmark_page'] = "true"
                else:
                    line['is_checkmark_page'] = "true"
                to_write.append(line)
            print(f"ann {ann} gets {len(to_write)} lines")
            writer = csv.DictWriter(f, fieldnames=to_write[0].keys())
            writer.writeheader() 
            writer.writerows(to_write)
                