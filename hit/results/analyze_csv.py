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
    x_before = x
    x = x.strip()
    # if False:
    # remove determiners in value ops 
    x = re.sub('" (a|an|the) ([^"]+) "', '" \g<2> "', x)
    # lowercase all value ops 
    x = re.sub('" ([^"]+) "', lambda m: m.group(0).lower(), x)
    # if x != x_before:
    # pdb.set_trace() 

    parsed_x = parse_lispress(x)
    return render_compact(parsed_x)


def read_csv(path, n_redundant=1):
    def process_row(line):
        manual_entry = line['Answer.manual_entry']
        if manual_entry.strip() == "" or manual_entry.strip() == "{}":
            radio_input = int(line['Answer.radio-input']) 

            # check if it's a list hit or not 
            if "Input.option_list" in line:
                option_list = json.loads(line['Input.option_list'])
                option_idx_list = json.loads(line['Input.option_idx_list'])
                chosen_tgt = option_list[radio_input-1]
                chosen_idx = option_idx_list[radio_input-1]
            else:
                chosen_tgt = line[f"Input.option_{radio_input - 1}"]
                chosen_idx = int(line[f"Input.option_{radio_input - 1}_idx"]) 
        else:
            chosen_tgt, chosen_idx = None, None
        line['chosen_tgt'] = chosen_tgt
        line['chosen_idx'] = chosen_idx
        line['manual_entry'] = manual_entry
        return line

    is_list = False
    with open(path) as f1:
        reader = csv.DictReader(f1)
        data = [process_row(x) for x in reader]
        if "Input.option_list" in data[0]:
            is_list = True
    # rows_by_example = defaultdict(list)
    rows_by_example = []
    for i in range(0, len(data), n_redundant):
        examples = data[i:i+n_redundant]
        rows_by_example.append(examples)
    return rows_by_example, is_list
 

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
    # print(out)
    # print(errs)
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
        return True, True, True, True
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

    dist_dict = defaultdict(lambda: {"dist": 0, "total": 0, "rewritten": 0, "selected": 0})
    for worker_id, worker_entries in entries_by_ann.items():
        for e in worker_entries:
            chosen_idx = e['chosen_idx']
            if chosen_idx is not None:
                dist_dict[worker_id]["selected"] += 1
            else:
                dist_dict[worker_id]["rewritten"] += 1
            if chosen_idx == "3": 
                dist_dict[worker_id]["dist"] += 1
            dist_dict[worker_id]["total"] += 1
    
    return dist_dict

def run_choose_and_rewrite(turk_data, 
                           json_data, 
                           args, 
                           aggregator="none", 
                           interact=False, 
                           rewrite_chosen=False, 
                           rewrite_all_baseline=False, 
                           is_list=False):
    from dataflow.core.utterance_tokenizer import UtteranceTokenizer
    tokenizer = UtteranceTokenizer()
    def tokenize(text):
        toked = tokenizer.tokenize(text)
        return " ".join(toked) 

    
    n_was_correct_now_incorrect = 0
    n_was_incorrect_now_correct = 0
    n_stayed_incorrect = 0
    n_stayed_correct = 0
    n_correct = 0
    n_correct_set = 0
    n_correct_most_likely = 0
    n_distractor = 0
    n_rewritten = 0
    total = 0
    total_gold_on_beam = 0
    rewritten = []
    non_rewritten_data = []
    rewritten_data = []
    # get turk_lookup 
    turk_lut = {}
    for turk_entry in turk_data: 
        te = turk_entry[0]
        te_str = ""
        if te['Input.user_turn_0'] != "":
            te_str = f"__User {te['Input.user_turn_0'].strip()} "
        if te['Input.agent_turn_0'] != "":
            te_str += f"__Agent {te['Input.agent_turn_0'].strip()} "
        if te['Input.user_turn_1'] != "":
            te_str += f"__User {te['Input.user_turn_1'].strip()}"

        te_str = re.sub('\\\\"', '"', te_str) 
        turk_lut[te_str] = turk_entry

    # for turk_entries, json_entry in zip(turk_data, json_data):
    for json_entry in json_data:
        try:
            key = re.sub("__StartOfProgram", "", json_entry['gold_src']).strip()
            turk_entries = turk_lut[key]
        except KeyError:
            print(f"Missing entry")
            # pdb.set_trace()
            continue
        # TODO (elias): need to implement three different cases 
        # case 1 where we take the majority vote 
            # in this case, what do we do with the rewrites? 
            # we can't take the majority vote of the rewritten examples,
            # so should we average?
            # How about: for cases where the majority is not rewrite, ignore rewrite 
            # For cases where the majority is rewrite, do rewrite for both/three and average   

        # case 2 where we count as success if any annotator solved the problem
        # case 3 where we take the mean of successes across annotators 
        # case 3 is actually the easiest 
        # aggregated_turk_entries = []
        aggregated_turk_entries_nonrewrite = []
        aggregated_turk_entries_rewrite = []
        if aggregator == "majority":
            # case 1: take majority vote across anns 
            chosen_tgt = [e['chosen_tgt'] for e in turk_entries]
            chosen_idx = [e['chosen_idx'] for e in turk_entries]
            # check if it has a majority 
            counts = Counter(chosen_tgt)
            # if majority decision is to rewrite, then do the rewrites 
            if None in counts.keys() and counts[None] > 1: 
                for e in turk_entries:
                    if e['chosen_tgt'] is None:
                        aggregated_turk_entries_rewrite.append(e)
            # otherwise, keep the majority decision 
            else:
                # get max by occurrences and select corresponding line 
                tgt_maj_vote = max(counts.items(), key=lambda x: x[1])[0]
                if tgt_maj_vote is None:
                    # there's a 3-way tie with None as one option, so just pick one other 
                    tgt_maj_vote = [k for k in counts.keys() if k is not None][0]
                maj_line = [e for e in turk_entries if e['chosen_tgt'] == tgt_maj_vote][0]
                if maj_line['chosen_idx'] is None:
                    pdb.set_trace()
                aggregated_turk_entries_nonrewrite.append(maj_line)

        elif aggregator == "none": 
            # case 3: take mean of successes across all anns 
            for turk_entry in turk_entries:
                chosen_turk_tgt = turk_entry['chosen_tgt']
                chosen_turk_idx = turk_entry['chosen_idx']
                if chosen_turk_idx == 3 and not is_list: 
                    # annotator chose the distractor, which is always index 4 
                    n_distractor += 1
                    continue 
                if chosen_turk_tgt is None or rewrite_chosen:
                    if rewrite_chosen and turk_entry is not None:
                        turk_entry['manual_entry'] = turk_entry['chosen_tgt']
                    aggregated_turk_entries_rewrite.append(turk_entry)
                elif rewrite_all_baseline:
                    turk_entry['manual_entry'] = turk_entry['Input.user_turn_1']
                    aggregated_turk_entries_rewrite.append(turk_entry)
                else:
                    aggregated_turk_entries_nonrewrite.append(turk_entry)

        for turk_entry in aggregated_turk_entries_nonrewrite:
            chosen_turk_tgt = turk_entry['chosen_tgt']
            chosen_turk_idx = turk_entry['chosen_idx']

            if chosen_turk_idx == 3 and not is_list: 
                # annotator chose the distractor, which is always index 4 
                n_distractor += 1
                continue 
            # get the corresponding lispress from json 
            chosen_lispress = json_entry['pred_tgts'][chosen_turk_idx]
            equivalent_lispress = [clean_lispress(x) for i, x in enumerate(json_entry['pred_tgts']) if json_entry['pred_translated'][i] == chosen_turk_tgt]
            # TODO (elias): sort these lispresses by their prob under the nucleus decode and take the top one 
            # they are already sorted from json creation, so can just take top 
            most_likely_equivalent_lispress = equivalent_lispress[0]

            # if len(equivalent_lispress) > 1:
                # pdb.set_trace()
            chosen_lispress = clean_lispress(chosen_lispress)
            chosen_by_nucleus_lispress = clean_lispress(json_entry['pred_tgts'][0])
            # get gold lisress 
            gold_lispress = json_entry['gold_tgt']
            gold_lispress = clean_lispress(gold_lispress)

            # TODO: (elias) add a new denominator here that only counts examples where the gold is in k-best 
            if chosen_lispress == gold_lispress:
                n_correct += 1
            if gold_lispress in equivalent_lispress:
                n_correct_set += 1
            if most_likely_equivalent_lispress == gold_lispress:
                n_correct_most_likely += 1
                if chosen_by_nucleus_lispress == gold_lispress:
                    n_stayed_correct += 1
                else:
                    n_was_incorrect_now_correct += 1
            else:
                if chosen_by_nucleus_lispress == gold_lispress:
                    # pdb.set_trace()
                    n_was_correct_now_incorrect += 1
                else:
                    n_stayed_incorrect += 1
                if interact and json_entry['bin'] == 0.75:
                    print(json_entry['bin'])
                    print(f"chosen idx {chosen_turk_idx}")
                    print(f"chosen gloss {chosen_turk_tgt}")
                    print(f"input: {json_entry['gold_src']}")
                    print(f"options: {json_entry['pred_translated']}")
                    print(f"chosen lispress: {chosen_lispress}")
                    print(f"gold: {gold_lispress}")
                    pdb.set_trace()
            options = json_entry['pred_tgts']
            options = [clean_lispress(o) for o in options]
            gold_on_beam = gold_lispress in options
            if gold_on_beam:
                total_gold_on_beam += 1

            if interact and float(json_entry['bin']) > 0.85:
                print(json_entry['bin'])
                print(f"gold on beam: {gold_on_beam}")
                if gold_on_beam:
                    print(f"\tgold: {json_entry['gold_tgt']}")
                    print(f"\ttop nucleus: {json_entry['pred_tgts'][0]}")
                    print(f"\tchosen: {chosen_lispress}")
                print()
                if gold_on_beam: 
                    pdb.set_trace()

            to_add = {"json_entry": json_entry, 
                     "turk_entry": turk_entry,
                     "chosen_lispress": chosen_lispress, 
                     "gold_lispress": gold_lispress, 
                     "gold_on_beam": gold_on_beam,
                     "is_correct": chosen_lispress == gold_lispress,
                     "is_correct_set": gold_lispress in equivalent_lispress,
                     "is_correct_most_likely": most_likely_equivalent_lispress == gold_lispress}
            non_rewritten_data.append(to_add)
            total += 1

        for turk_entry  in aggregated_turk_entries_rewrite:
            # get user context 
            gold_src = split_source(json_entry['gold_src'])
            manual_entry = tokenize(turk_entry['manual_entry'])
            if len(gold_src) == 1:
                src_str = f"__User {manual_entry} __StartOfProgram"
            else:
                src_str = f"__User {gold_src[0]} __Agent {gold_src[1]} __User {manual_entry} __StartOfProgram"
            rewritten.append((src_str, json_entry['gold_tgt']))
            n_rewritten += 1 
            to_add = {"json_entry": json_entry,
                     "turk_entry": turk_entry}
            rewritten_data.append(to_add)

    print(f"Rewritten: {n_rewritten}")
    # decode rewritten examples
    print(f"decoding {len(rewritten)} examples from {args.checkpoint_dir}")
    rewritten_n_correct = 0
    rewritten_total = 0
    miso_rewritten_lispress = decode(rewritten, args.checkpoint_dir)
    rewritten_inputs, gold_tgts = zip(*rewritten)
    for i, (rewritten_input, miso_rewritten_lispress, gold_tgt) in enumerate(zip(rewritten_inputs, miso_rewritten_lispress, gold_tgts)):
        gold_tgt = clean_lispress(gold_tgt)

        if miso_rewritten_lispress == gold_tgt:
            rewritten_n_correct += 1
        else:
            if interact:
                pdb.set_trace()
        rewritten_total += 1

        rewritten_data[i]['rewritten'] = miso_rewritten_lispress
        rewritten_data[i]['is_correct'] = miso_rewritten_lispress == gold_tgt
        rewritten_data[i]['is_correct_set'] = miso_rewritten_lispress == gold_tgt

    print(f"Dynamic breakdown (ignoring rewritten): {n_stayed_incorrect} stayed incorrect, \
                                {n_was_correct_now_incorrect} were correct now incorrect, \
                                {n_stayed_correct} stayed correct, \
                                {n_was_incorrect_now_correct} were incorrect now correct")

    if total > 0:
        print(f"Accuracy (non-rewritten): {n_correct}/{total}: \
                {n_correct / total*100:.2f}%")
        print(f"Accuracy - in set (non-rewritten): {n_correct_set}/{total}: \
                {n_correct_set / total*100:.2f}%")
        print(f"Accuracy - most likely in set (non-rewritten): {n_correct_most_likely}/{total}: \
                {n_correct_most_likely / total*100:.2f}%")
        print(f"Accuracy (non-rewritten, gold on beam): {n_correct}/{total_gold_on_beam}: \
                {n_correct / total_gold_on_beam*100:.2f}%")
    if rewritten_total > 0:
        print(f"Accuracy (rewritten): {rewritten_n_correct}/{rewritten_total}: \
                {rewritten_n_correct / rewritten_total*100:.2f}%")

    combo_n_correct = n_correct + rewritten_n_correct
    combo_total = total + rewritten_total
    if combo_total > 0:
        print(f"Accuracy (combined): {combo_n_correct}/{combo_total}: \
                {combo_n_correct / combo_total *100:.2f}%")

    return non_rewritten_data, rewritten_data

def run_iaa(turk_data, json_data):
    ann_scores = annotator_scores(turk_data)
    ann_scores = {k: {k2: v[k2] / v["total"] 
                            if k2 != "total" else v[k2] 
                            for k2 in v.keys() } 
                for k, v in ann_scores.items()}

    print("percentage of time each annotator chose the distractor")
    for ann, ann_data in ann_scores.items():
        print_str = [f"{k}: {v * 100:.2f}%" for k, v in ann_data.items() if k != "total"]
        print_str.append(f"total: {ann_data['total']}")
        print_str = " ".join(print_str)
        print(f"{ann}: {print_str}")
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
    print(f"all annotators agree on target chosen:  {iaa_scores['target_equal']} / {target_equal_total} = {target_equal_agreement*100:.2f}%")
    print(f"all annotators agree on index chosen: {iaa_scores['idx_equal']} / {idx_equal_total} =  {idx_equal_agreement*100:.2f}%")
    print()
    print(f"{iaa_scores['any_rewrite']} examples have at least one rewritten sentence")
    print(f"all annotators rewrote the example: {iaa_scores['all_rewrite']} / {iaa_scores['any_rewrite']} = {all_rewritten_agreement*100:.2f}%")

    maj_rewritten_agreement = iaa_maj_scores["all_rewrite"] / iaa_maj_scores['any_rewrite']
    target_maj_agreement = iaa_maj_scores["target_equal"] / target_maj_total
    idx_maj_agreement = iaa_maj_scores["idx_equal"] / idx_maj_total
    print()
    print(f"{target_maj_total} examples have a majority of non-rewritten sentences")
    print(f"a majority of annotators agree on target chosen: {iaa_maj_scores['target_equal']} / {target_maj_total} = {target_maj_agreement*100:.2f}%")
    print(f"a majority annotators agree on index chosen: {iaa_maj_scores['idx_equal']} / {idx_maj_total} = {idx_maj_agreement*100:.2f}%")
    print()
    print(f"{iaa_maj_scores['any_rewrite']} examples have at least one rewritten sentence")
    print(f"a majority of annotators rewrote the example: {iaa_maj_scores['all_rewrite']} / {iaa_maj_scores['any_rewrite']} = {maj_rewritten_agreement*100:.2f}%")

def main(args):
    print(f"Reading data from {args.csv}")
    turk_data, is_list = read_csv(args.csv, n_redundant=args.n_redundant)

    print(f"Reading data from {args.json}")
    json_data = read_json(args.json)

    if args.do_iaa:
        run_iaa(turk_data, json_data)

    if args.do_rewrites:
        run_choose_and_rewrite(turk_data, 
                                json_data, 
                                args, 
                                aggregator=args.aggregator, 
                                interact=args.interact, 
                                rewrite_chosen=args.rewrite_chosen, 
                                rewrite_all_baseline=args.rewrite_all_baseline,
                                is_list=is_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="csv output from mturk")
    parser.add_argument("--json", type=str, required=True, help="json input that generated the csv input for mturk")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoint dir for miso model", default="/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data")
    parser.add_argument("--do_rewrites", action="store_true", help="whether to rewrite examples")
    parser.add_argument("--rewrite_chosen", action="store_true", help="whether to rewrite the chosen sentence instead of taking the parse associated with it")
    parser.add_argument("--rewrite_all_baseline", action="store_true", help="set if you want to rewrite the original input sentences, as a baseline for rewrite_chosen")
    parser.add_argument("--do_iaa", action="store_true", help="whether to run iaa")
    parser.add_argument("--aggregator", type=str, default="none", help="aggregator to use for rewrites", choices = ["none", "majority"])
    parser.add_argument("--n_redundant", type=int, default=1)
    parser.add_argument("--interact", action="store_true")
    args = parser.parse_args() 
    main(args)