import argparse 
import json 
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.bin_file, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    out_dir =  Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    stemname = Path(args.bin_file).stem
    with open(out_dir / f"{stemname}.src_tok", "w") as src_f,\
        open(out_dir / f"{stemname}.tgt", "w") as tgt_f, \
        open(out_dir / f"{stemname}.bins", "w") as bin_f, \
        open(out_dir / f"{stemname}.idx", "w") as idx_f: 
        for d in data:
            src_f.write(d["source"].strip() + "\n")
            tgt_f.write(d["target"].strip() + "\n")
            bin_f.write(f"{d['bin']:.2f}" + "\n")
            idx_f.write(f"{d['index']}" + "\n")