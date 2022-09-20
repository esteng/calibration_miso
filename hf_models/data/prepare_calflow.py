import json
import pathlib
import argparse

def main(args):
    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for split in ["train", "dev_valid", "test_valid", "valid", "test"]: 
        with open(data_dir.joinpath(f"{split}.src_tok")) as src_f, \
            open(data_dir.joinpath(f"{split}.tgt")) as tgt_f, \
            open(data_dir.joinpath(f"{split}.datum_id")) as id_f, \
            open(out_dir.joinpath(f"{split}.jsonl"), "w") as out_f:
            for src, tgt, id in zip(src_f, tgt_f, id_f):
                src = src.strip()
                tgt = tgt.strip()
                tgt = "<extra_id_0> " + tgt + " <extra_id_1>"
                id = json.loads(id.strip())
                datum = {"src": src, "tgt": tgt, "id": id}
                datum = json.dumps(datum)
                out_f.write(datum + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help = "Path to smcalflow.agents.data")
    parser.add_argument("--out_dir", type=str, required=True, help = "Path to output jsonl files")
    args = parser.parse_args()
    main(args)