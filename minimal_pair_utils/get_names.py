import argparse
import json 
import pdb 

from tqdm import tqdm 
from dataflow.core.lispress import parse_lispress, lispress_to_program
from dataflow.core.program import ValueOp, BuildStructOp


def get_names(args):
    names = set()
    with open(args.target_path) as f1:
        for line in tqdm(f1):
            p, _= lispress_to_program(parse_lispress(line.strip()),0)
            for expr in p.expressions:
                if isinstance(expr.op, BuildStructOp) and expr.op.op_schema == "PersonName.apply":
                    child_idx = expr.arg_ids[0]
                    child_expr = [x for x in p.expressions if x.id == child_idx][0]
                    data = json.loads(child_expr.op.value)
                    names.add(data['underlying'].strip())
                    #print(f"added {data['underlying'].strip()}")
    return list(names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-path", type=str, required=True, help="path to .tgt file")
    parser.add_argument("--out-path", type=str, required=True, help="output json destination")
    args = parser.parse_args()

    name_list = get_names(args)
    with open(args.out_path,"w") as f1:
        json.dump(name_list, f1)