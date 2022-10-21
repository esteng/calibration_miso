# get input just for stratified sample to reduce computation time 
from genericpath import exists
import json 
from pathlib import Path
import re 

path_to_strat_three_line = "hit/data/for_hit_round_4/stratified_data_by_bin.tgt"
with open(path_to_strat_three_line) as f1:
    strat_three_lines = [json.loads(x) for x in f1.readlines()]
    strat_three_lines = [(x['src_str'].strip(), x['midpoint']) for x in strat_three_lines]

data_path = Path("/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data.from_benchclamp/")

src_file = data_path / "dev_all.src_tok"
tgt_file = data_path / "dev_all.tgt"
idx_file = data_path / "dev_all.idx"

with open(src_file) as f1:
    src_lines = f1.readlines()
    src_lines = [re.sub("__StartOfProgram", "", x).strip() for x in src_lines]
with open(tgt_file) as f1:
    tgt_lines = [x.strip() for x in f1.readlines()]
with open(idx_file) as f1:
    idx_lines = [x.strip() for x in f1.readlines()]


data_by_src = {}
for line, bin in strat_three_lines:
    for src, tgt, idx in zip(src_lines, tgt_lines, idx_lines):
        if line == src.strip():
            data_by_src[src] = {"src": src, "tgt": tgt, "idx": idx, "bin": bin}

print(len(data_by_src))
out_dir = Path("hit/data/for_miso/stratified_round_4")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "stratified.src_tok", "w") as src_f, \
    open(out_dir / "stratified.tgt", "w") as tgt_f,\
    open(out_dir / "stratified.idx", "w") as idx_f,\
    open(out_dir / "stratified.bins", "w") as bin_f:
    for src in data_by_src:
        src_f.write(data_by_src[src]["src"] + " __StartOfProgram\n")
        tgt_f.write(data_by_src[src]["tgt"] + "\n")
        idx_f.write(data_by_src[src]["idx"] + "\n")
        bin_f.write(f"{data_by_src[src]['bin']}\n")



