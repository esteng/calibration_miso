import argparse
from collections import defaultdict, Counter
import csv 
import pathlib
import json
import re 
import os 
import pdb 
import subprocess 
import numpy as np

from dataflow.core.lispress import parse_lispress, render_compact

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("hit/scripts/")))
# from hit.scripts.prep_for_translate import split_source
from prep_for_translate import split_source

def clean_lispress(x):
    x = x.strip()
    parsed_x = parse_lispress(x)
    return render_compact(parsed_x)


def read_csv(path, n_redundant=1):
    def process_row(line):
        manual_entry = line['Answer.manual_entry']
        if manual_entry.strip() == "" or manual_entry.strip() == "{}":
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
    # rows_by_example = defaultdict(list)
    rows_by_example = []
    for i in range(0, len(data), n_redundant):
        examples = data[i:i+n_redundant]
        rows_by_example.append(examples)
        # rows_by_example[i//n_redundant] = examples
        # rows_by_turker[row['WorkerId']].append(row)
    return rows_by_example
    # return rows_by_turker   

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

def get_pairwise_iaa_scores(turk_entries): 
    # TODO (elias): fix this metric 
    done = []
    all_rewrite_equal = []
    all_tgts_equal = []
    all_idxs_equal = []
    n_rewrite, n_tgt, n_idx = 0, 0, 0

    each_is_rewrite = np.array([e['manual_entry'].strip() not in ["{}", ""] 
                        for e in turk_entries]) 

    for i in range(len(turk_entries)):
        for j in range(len(turk_entries)):
            if i == j: 
                continue
            if (i,j) in done or (j,i) in done: 
                continue 

            line_a = turk_entries[i]
            line_b = turk_entries[j]
            # if the pair is a rewrite, add to all rewrite equal 
            if line_a['manual_entry'] not in ["{}", ""] \
                and line_b['manual_entry'] not in ["{}", ""]:
                all_rewrite_equal.append(1)
                n_rewrite += 1
            # if one is a rewrite and the other is not, then that counts against 
            elif line_a['manual_entry'] not in ["{}", ""] \
                and line_b['manual_entry']  in ["{}", ""] or \
                line_a['manual_entry'] in ["{}", ""] \
                and line_b['manual_entry'] not in ["{}", ""]:
                all_rewrite_equal.append(0)
                n_rewrite += 1
            # if neither is a rewrite, then check if the chosen tgt is the same 
            else:
                all_tgts_equal.append(int(line_a['chosen_tgt'] == line_b['chosen_tgt']))
                all_idxs_equal.append(int(line_a['chosen_idx'] == line_b['chosen_idx']))
                n_tgt += 1
                n_idx += 1
            done.append((i,j))
    rewrite_equal = np.mean(all_rewrite_equal)
    if np.isnan(rewrite_equal):
        rewrite_equal = 0
    tgts_equal = np.mean(all_tgts_equal)
    if np.isnan(tgts_equal):
        tgts_equal = 0
    idxs_equal = np.mean(all_idxs_equal)
    if np.isnan(idxs_equal):
        idxs_equal = 0
    return rewrite_equal, n_rewrite, tgts_equal, n_tgt, idxs_equal, n_idx
   

def get_iaa_scores(turk_entries, majority=False): 
    def all_equal(list):
        return len(set(list)) == 1
    def has_majority(list):
        list_counter = Counter(list)
        max_val = max(list_counter.values())
        return max_val > len(list) / 2

    # get iaa scores for a group of turk entries
    if len(turk_entries) == 1:
        return True, True, True
    else: 
        each_is_rewrite = np.array([e['manual_entry'].strip() not in ["{}", ""] 
                            for e in turk_entries]) 
        if majority:
            all_rewrite = np.mean(each_is_rewrite) > 0.5
        else:
            all_rewrite = all(each_is_rewrite)
        any_rewrite = any(each_is_rewrite)
        # if there are at least 2 non-rewritten examples, we can compute this metric 
        if sum(~each_is_rewrite) > 2:
            all_choices_tgt = [turk_entry['chosen_tgt'] for turk_entry in turk_entries]
            all_choices_idx = [turk_entry['chosen_idx'] for turk_entry in turk_entries]
            if majority:
                target_equal = int(has_majority(all_choices_tgt))
                idx_equal = int(has_majority(all_choices_idx))
            else:
                target_equal = int(all_equal(all_choices_tgt))
                idx_equal = int(all_equal(all_choices_idx)) 
            
        else:
            # if there are fewer than 2 non-rewritten examples, the metric doesn't make sense
            target_equal = None
            idx_equal = None
        return float(all_rewrite), int(any_rewrite), target_equal, idx_equal

def annotator_scores(turk_data):
    entries_by_ann = defaultdict(list)
    for turk_entries in turk_data: 
        for turk_entry in turk_entries:
            entries_by_ann[turk_entry['WorkerId']].append(turk_entry)

    dist_dict = defaultdict(lambda: {"dist": 0, "total": 0})
    for worker_id, worker_entries in entries_by_ann.items():
        for e in worker_entries:
            chosen_idx = e['chosen_idx']
            if chosen_idx == "3": 
                dist_dict[worker_id]["dist"] += 1
            dist_dict[worker_id]["total"] += 1
    
    return dist_dict

def run_choose_and_rewrite(turk_data, json_data):
    from dataflow.core.utterance_tokenizer import UtteranceTokenizer
    tokenizer = UtteranceTokenizer()
    def tokenize(text):
        toked = tokenizer.tokenize(text)
        return " ".join(toked) 

    n_correct = 0
    n_distractor = 0
    n_rewritten = 0
    total = 0
    rewritten = []
    for turk_entries, json_entry in zip(turk_data, json_data):
        for turk_entry in turk_entries:
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
                manual_entry = tokenize(turk_entry['manual_entry'], tokenizer)
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

def run_iaa(turk_data, json_data):
    ann_scores = annotator_scores(turk_data)
    ann_scores = {k: v["dist"] / v["total"] for k, v in ann_scores.items()}
    ann_scores = {k: f"{v * 100:.2f}%" for k, v in ann_scores.items()}
    print("percentage of time each annotator chose the distractor")
    print(ann_scores)

    iaa_scores = {"all_rewrite": 0, "any_rewrite": 0, "target_equal": 0, "idx_equal": 0}
    iaa_maj_scores = {"all_rewrite": 0, "any_rewrite": 0, "target_equal": 0, "idx_equal": 0}
    pw_iaa_scores = {"rewrite": 0, "rewrite_total": 0, "target": 0, "target_total":0,  "idx": 0, "idx_total": 0}
    target_equal_total, idx_equal_total = 0, 0
    target_maj_total, idx_maj_total = 0, 0
    for turk_entries, json_entry in zip(turk_data, json_data):
        all_rewrite, any_rewrite, target_equal, idx_equal = get_iaa_scores(turk_entries)
        iaa_scores["all_rewrite"] += all_rewrite
        iaa_scores["any_rewrite"] += any_rewrite
        if target_equal is not None:
            target_equal_total += 1
            iaa_scores["target_equal"] += target_equal
        if idx_equal is not None:
            idx_equal_total += 1
            iaa_scores["idx_equal"] += idx_equal
        maj_rewrite, __, target_maj, idx_maj = get_iaa_scores(turk_entries, majority=True)
        iaa_maj_scores["all_rewrite"] += maj_rewrite 
        iaa_maj_scores["any_rewrite"] += any_rewrite
        if target_maj is not None:
            target_maj_total += 1
            iaa_maj_scores["target_equal"] += target_maj 
        if idx_maj is not None:
            idx_maj_total += 1
            iaa_maj_scores["idx_equal"] += idx_maj


    all_rewritten_agreement = iaa_scores["all_rewrite"] / iaa_scores['any_rewrite']
    target_equal_agreement = iaa_scores["target_equal"] / target_equal_total
    idx_equal_agreement = iaa_scores["idx_equal"] / idx_equal_total
    print()
    print(f"{target_equal_total} examples have at least two non-rewritten sentences")
    print(f"all annotators agree on target chosen: {target_equal_agreement*100:.2f}%")
    print(f"all annotators agree on index chosen: {idx_equal_agreement*100:.2f}%")
    print()
    print(f"{iaa_scores['any_rewrite']} examples have at least one rewritten sentence")
    print(f"all annotators rewrote the example: {all_rewritten_agreement*100:.2f}%")

    maj_rewritten_agreement = iaa_maj_scores["all_rewrite"] / iaa_maj_scores['any_rewrite']
    target_maj_agreement = iaa_maj_scores["target_equal"] / target_maj_total
    idx_maj_agreement = iaa_maj_scores["idx_equal"] / idx_maj_total
    print()
    print(f"{target_maj_total} examples have a majority of non-rewritten sentences")
    print(f"a majority of annotators agree on target chosen: {target_maj_agreement*100:.2f}%")
    print(f"a majority annotators agree on index chosen: {idx_maj_agreement*100:.2f}%")
    print()
    print(f"{iaa_maj_scores['any_rewrite']} examples have at least one rewritten sentence")
    print(f"a majority of annotators rewrote the example: {maj_rewritten_agreement*100:.2f}%")

    # pw_rewrite_score = pw_iaa_scores["rewrite"] / pw_iaa_scores["rewrite_total"]
    # pw_target_score = pw_iaa_scores["target"] / pw_iaa_scores["target_total"]
    # pw_idx_score = pw_iaa_scores["idx"] / pw_iaa_scores["idx_total"]
    # print()
    # print(f"pairwise agreement on rewrite: {pw_rewrite_score*100:.2f}%")
    # print(f"pairwise agreement on target: {pw_target_score*100:.2f}%")
    # print(f"pairwise agreement on idx: {pw_idx_score*100:.2f}%")

def main(args):
    print(f"Reading data from {args.csv}")
    turk_data = read_csv(args.csv, n_redundant=args.n_redundant)
    print(f"Reading data from {args.json}")
    json_data = read_json(args.json)

    if args.do_iaa:
        run_iaa(turk_data, json_data)

    if args.do_rewrites:
        run_choose_and_rewrite(turk_data, json_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="csv output from mturk")
    parser.add_argument("--json", type=str, required=True, help="json input that generated the csv input for mturk")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoint dir for miso model", default="/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data")
    parser.add_argument("--do_rewrites", action="store_true", help="whether to rewrite examples")
    parser.add_argument("--do_iaa", action="store_true", help="whether to run iaa")
    parser.add_argument("--n_redundant", type=int, default=1)
    args = parser.parse_args() 
    main(args)