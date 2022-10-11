import argparse 
import csv 
import pathlib
import json
import re 
import os 
import pdb 
import subprocess 
import en_core_web_sm

from dataflow.core.lispress import parse_lispress, render_compact
from dataflow.core.utterance_tokenizer import UtteranceTokenizer
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("hit/scripts/")))
# from hit.scripts.prep_for_translate import split_source
from prep_for_translate import split_source

tokenizer = UtteranceTokenizer()
def tokenize(text):
    toked = tokenizer.tokenize(text)
    return " ".join(toked) 

def clean_lispress(x):
    x = x.strip()
    parsed_x = parse_lispress(x)
    return render_compact(parsed_x)

def read_csv(path):
    def process_row(line):
        manual_entry = line['Answer.manual_entry']
        if manual_entry.strip() == "":
            radio_input = int(line['Answer.radio-input'])
            chosen_tgt = line[f"Input.option_{radio_input - 1}"]
            chosen_idx = int(line[f"Input.option_{radio_input - 1}_idx"]) - 1
        else:
            chosen_tgt, chosen_idx = None, None
        line['chosen_tgt'] = chosen_tgt
        line['chosen_idx'] = chosen_idx
        line['manual_entry'] = manual_entry
        return line 

    with open(path) as f1:
        reader = csv.DictReader(f1)
        data = [process_row(x) for x in reader]
    return data 

def read_json(path): 
    with open(path) as f1:
        data = [json.loads(x) for x in f1]
    return data

def decode(rewritten, checkpoint_dir):
    # write everything to a temp file 
    tempfile_src = pathlib.Path(__file__).parent / "temp.src_tok"
    tempfile_tgt = pathlib.Path(__file__).parent / "temp.tgt"
    with open(tempfile_src,"w") as f1, open(tempfile_tgt,"w") as f2:
        for src, tgt in rewritten:
            f1.write(src + "\n")
            f2.write(tgt + "\n")

    # run the decoding script
    decode_command = ["sh", "/home/estengel/incremental-function-learning/experiments/calflow.sh", "-a", "eval_fxn"]
    env = os.environ.copy()
    # print(temp_file)
    env['CHECKPOINT_DIR'] = checkpoint_dir
    env['TEST_DATA'] = str(tempfile_src.parent / "temp")
    env['FXN'] = "none"
    p = subprocess.Popen(decode_command, stdout=subprocess.PIPE, env=env)
    out, errs = p.communicate()
    # out = out.decode('utf-8')
    # errs = errs.decode('utf-8')
    out_file = pathlib.Path(checkpoint_dir) / "translate_output" / f"temp.tgt"
    out_lispress = []
    with open(out_file) as f1:
        for l in f1.readlines():
            lispress = clean_lispress(l)
            out_lispress.append(lispress)
    return out_lispress


def main(args):
    turk_data = read_csv(args.csv)
    json_data = read_json(args.json)

    n_correct = 0
    n_distractor = 0
    n_rewritten = 0
    total = 0
    rewritten = []
    for turk_entry, json_entry in zip(turk_data, json_data):
        chosen_turk_tgt = turk_entry['chosen_tgt']
        chosen_turk_idx = turk_entry['chosen_idx']
        if chosen_turk_idx == 3: 
            # annotator chose the distractor, which is always index 4 
            n_distractor += 1
            continue 

        if chosen_turk_tgt is None:
            # manual entry, pass for now 
            # print(f"Manual entry: {turk_entry['manual_entry']}")
            # get user context 
            gold_src = split_source(json_entry['gold_src'])
            manual_entry = tokenize(turk_entry['manual_entry'])
            if len(gold_src) == 1:
                src_str = f"__User {manual_entry} __StartOfProgram"
            else:
                src_str = f"__User {gold_src[0]} __Agent {gold_src[1]} __User {manual_entry} __StartOfProgram"
            rewritten.append((src_str, json_entry['gold_tgt']))
            n_rewritten += 1 
            continue 
        # get the corresponding lispress from json 
        try:
            chosen_lispress = json_entry['pred_tgts'][chosen_turk_idx]
        except IndexError:
            pdb.set_trace()
        if chosen_lispress is None:
            # fence example
            # this shouldn't actually happen
            # skip for now, but later should raise error 
            continue 
        chosen_lispress = clean_lispress(chosen_lispress)
        # get gold lisress 
        gold_lispress = json_entry['gold_tgt']
        gold_lispress = clean_lispress(gold_lispress)

        # print(f"Chosen tgt: {chosen_turk_tgt}")
        # print(f"Chosen lispress: {chosen_lispress}") 
        # print(f"Gold lispress: {gold_lispress}")
        if chosen_lispress == gold_lispress:
            n_correct += 1
        total += 1

    print(f"Rewritten: {n_rewritten}")

    # decode rewritten examples
    print(f"decoding {len(rewritten)} examples from {args.checkpoint_dir}")
    rewritten_n_correct = 0
    rewritten_total = 0
    miso_rewritten_lispress = decode(rewritten, args.checkpoint_dir)
    rewritten_inputs, gold_tgts = zip(*rewritten)
    for rewritten_input, miso_rewritten_lispress, gold_tgt in zip(rewritten_inputs, miso_rewritten_lispress, gold_tgts):
        gold_tgt = clean_lispress(gold_tgt)
        # print(rewritten_input)
        # print(miso_rewritten_lispress)
        # print(gold_tgt)
        # pdb.set_trace()
        if miso_rewritten_lispress == gold_tgt:
            rewritten_n_correct += 1
        rewritten_total += 1

    print(f"Accuracy (non-rewritten): {n_correct}/{total}: \
            {n_correct / total*100:.2f}%")
    print(f"Accuracy (rewritten): {rewritten_n_correct}/{rewritten_total}: \
            {rewritten_n_correct / rewritten_total*100:.2f}%")

    combo_n_correct = n_correct + rewritten_n_correct
    combo_total = total + rewritten_total
    print(f"Accuracy (combined): {combo_n_correct}/{combo_total}: \
            {combo_n_correct / combo_total *100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="csv output from mturk")
    parser.add_argument("--json", type=str, required=True, help="json input that generated the csv input for mturk")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoint dir for miso model", default="/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data")
    args = parser.parse_args() 
    main(args)