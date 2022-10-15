import csv 
import pathlib

res_path = pathlib.Path("hit/results/round2_incomplete") 
input_paths = ["hit/data/for_hit/from_stratified/for_turk_round2/exclude_A2LMQ4497NMK3S.csv",
              "hit/data/for_hit/from_stratified/for_turk_round2/exclude_A2RBF3IIJP15IH.csv"]

out_path = "hit/data/for_hit/from_stratified/for_turk_round2/for_AKQAI78JTXXC9.csv"

input_data = []
for path in input_paths:
    with open(path) as f1:
        reader = csv.DictReader(f1)
        data = list(reader)
        input_data += data

res_data_by_key = {}
for path in res_path.glob("Batch*.csv"):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        for line in data:
            key = (line['Input.user_turn_0'].strip(), line["Input.agent_turn_0"].strip(), line['Input.user_turn_1'].strip()) 
            res_data_by_key[key] = line

written = 0
with open(out_path, "w") as f:
    writer = csv.DictWriter(f, fieldnames=input_data[0].keys())
    writer.writeheader()
    for line in input_data: 
        key = (line['user_turn_0'].strip(), line["agent_turn_0"].strip(), line['user_turn_1'].strip()) 
        if key not in res_data_by_key.keys(): 
            writer.writerow(line)
            written += 1

print(written)
assert(written == 19)

