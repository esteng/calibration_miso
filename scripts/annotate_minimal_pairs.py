import pathlib
import argparse 
from dataflow.core.lispress import parse_lispress, render_pretty, render_compact

class Example:
    def __init__(self,
                 input,
                 plans):
        self.input = input
        self.plans = plans
        self.lispress = [parse_lispress(p) for p in plans]

def get_existing_examples(input_path, output_path, fxn_of_interest, num):
    with open(input_path) as inp, open(output_path) as out:
        in_lines = [x.strip() for x in inp.readlines()] 
        out_lines = [x.strip() for x in out.readlines()] 
    examples = [Example(inl.strip(), [outl.strip()]) for inl, outl in zip(in_lines, out_lines) if fxn_of_interest in outl.strip().split(" ")]
    return examples[0:num]

def write_example(ex, id, out_path):
    to_write_path = out_path.joinpath(id)
    to_write_path.mkdir(exist_ok=True, parents=True)
    with open(to_write_path.joinpath(f"{id}.src"),'w') as src,\
         open(to_write_path.joinpath(f"{id}.tgt"), 'w') as tgt: 
         src.write(ex.input)
         tgt.write(ex.plans[0])


def annotate_all(examples, out_path):
    new_examples = []
    for i, ex in enumerate(examples):
        annotations = annotate_one(ex, i, out_path)
        new_examples += annotations
    return new_examples 

def get_user_data(gold_example):
    user_input = input("New input sent: \n")
    print()
    print(render_compact(gold_example.lispress[0]))
    user_output = input("New output sent: \n")
    print()
    print(f"Input: {user_input}")
    print(f"Output: {user_output}")
    return user_input, user_output

def annotate_one(gold_example, gold_idx, out_path): 
    annotations = []
    print(gold_example.input)
    print(gold_example.plans[0])

    continue_code = input("Keep annotating example? [Y/n]: ")
    ex_idx = 0
    while continue_code.strip() in ['y','Y','']:
        user_input, user_output = get_user_data(gold_example)
        confirm_code = input("Confirm? [Y/n]: ")
        if confirm_code.strip() in ['y','Y','']:
            pass  
        else:
            continue_code = input("Keep annotating example? [Y/n]: ")
            if continue_code == "n":
                break
            user_input, user_output = get_user_data(gold_example)

        try:
            user_lispress = parse_lispress(user_output)
        except ValueError:
            print(f"Invalid program: {user_output}")
            continue_code = input("Keep annotating example? [Y/n]: ")
            if continue_code == "n":
                break
            user_input, user_output = get_user_data(gold_example)
        print("Making example!") 
        user_example = Example(user_input, [user_output])
        annotations.append(user_example)
        example_id = f"{gold_idx}_{ex_idx}"
        write_example(user_example, example_id, out_path)
        ex_idx += 1
        continue_code = input("Keep annotating example? [Y/n]: ")
    return annotations  


def main(args): 
    train_path = pathlib.Path(args.train_path)
    src_path = train_path.joinpath("train.src_tok")
    tgt_path = train_path.joinpath("train.tgt")
    examples = get_existing_examples(src_path, tgt_path, args.fxn_of_interest, args.num)
    out_path = pathlib.Path(args.out_path)
    new_examples = annotate_all(examples, out_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="/home/t-eliass//resources/data/smcalflow.agent.data/")
    parser.add_argument("--num",type=int, default=100)
    parser.add_argument("--fxn-of-interest", type=str, default="FindManager")
    parser.add_argument("--out-path", type=str, default="/home/t-eliass//resources/data/smcalflow_augmented/FindManager/")

    args = parser.parse_args() 

    main(args)