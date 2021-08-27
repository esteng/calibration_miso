import argparse
from pathlib import Path 

def read_data(path, use_test = False): 
    path = Path(path) 
    if not use_test:
        src_path = path.joinpath("train.src_tok") 
        idx_path = path.joinpath("train.idx") 
        tgt_path = path.joinpath("train.tgt") 
    else:
        src_path = path.joinpath("test_valid.src_tok") 
        idx_path = path.joinpath("test_valid.idx") 
        tgt_path = path.joinpath("test_valid.tgt") 

    with open(src_path) as f1:
        src_lines = [line.strip().split(" ") for line in f1.readlines()]
    with open(tgt_path) as f1:
        tgt_lines = [line.strip().split(" ") for line in f1.readlines()]
    with open(idx_path) as f1:
        idx_lines = [int(line.strip()) for line in f1.readlines()]
    return src_lines, idx_lines, tgt_lines

def subset(train_src, train_idx, train_tgt, fxn_of_interest):
    fxn_src, fxn_idx, fxn_tgt = [], [], []
    for src, idx, tgt in zip(train_src, train_idx, train_tgt): 
        if fxn_of_interest in tgt: 
            fxn_src.append(src)
            fxn_idx.append(idx)
            fxn_tgt.append(tgt)
    return fxn_src, fxn_idx, fxn_tgt

def main(args):
    train_src, train_idx, train_tgt = read_data(args.train_path, use_test = args.use_test)
    fxn_src, fxn_idx, fxn_tgt = subset(train_src, train_idx, train_tgt, args.fxn_of_interest)

    if not args.use_test:
        with open(Path(args.write_path).joinpath("fxn_train.src_tok"), "w") as src_f,\
            open(Path(args.write_path).joinpath("fxn_train.idx"), "w") as idx_f,\
            open(Path(args.write_path).joinpath("fxn_train.tgt"), "w") as tgt_f: 
            for src, idx, tgt in zip(fxn_src, fxn_idx, fxn_tgt): 
                src_f.write(" ".join(src) + "\n") 
                tgt_f.write(" ".join(tgt) + "\n") 
                idx_f.write(f"{idx}\n")
    else:
        with open(Path(args.write_path).joinpath("fxn_test_valid.src_tok"), "w") as src_f,\
            open(Path(args.write_path).joinpath("fxn_test_valid.idx"), "w") as idx_f,\
            open(Path(args.write_path).joinpath("fxn_test_valid.tgt"), "w") as tgt_f: 
            for src, idx, tgt in zip(fxn_src, fxn_idx, fxn_tgt): 
                src_f.write(" ".join(src) + "\n") 
                tgt_f.write(" ".join(tgt) + "\n") 
                idx_f.write(f"{idx}\n")

    print(f"wrote to {args.write_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True, help="path to train data")
    parser.add_argument("--write-path", type=str, required=True, help = "path to write fxn output")
    parser.add_argument("--fxn-of-interest", type=str, required=True, help = "fxn to subset by")
    parser.add_argument("--use-test", action="store_true", help="use test set instead of train") 
    args = parser.parse_args() 
    main(args)
