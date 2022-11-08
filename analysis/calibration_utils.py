from typing import Tuple
import json
import re 
import numpy as np 
from collections import defaultdict

from dataflow.core.lispress import parse_lispress, render_compact
from dataflow.core.turn_prediction import TurnPrediction, TurnAnswer
from dataflow.core.dialogue import TurnId, ProgramExecutionOracle
from dataflow.core.lispress import try_round_trip

def evaluate_prediction_exact_match(
    pred: TurnPrediction, gold: TurnAnswer
) -> Tuple[bool, bool]:
    assert pred.datum_id == gold.datum_id, f"mismatched data: {pred}, {gold}"
    pred_lispress = try_round_trip(pred.lispress)
    gold_lispress = try_round_trip(gold.lispress)
    # if pred_lispress != gold_lispress:
    #     print(
    #         f"Misprediction on {gold.datum_id.dialogue_id}:{gold.datum_id.turn_index} | {gold.user_utterance}\nPred: {pred_lispress}\nGold: {gold_lispress}\n"
    #     )
    # elif not gold.program_execution_oracle.refer_are_correct:
    #     print(
    #         f"Example {gold.datum_id.dialogue_id}:{gold.datum_id.turn_index} can't be correct because the refer call is not correct.\n"
    #     )
    return (
        pred_lispress == gold_lispress
        and gold.program_execution_oracle.refer_are_correct,
        pred_lispress == gold_lispress,
    )

def single_exact_match(pred_lispress, gold_lispress):
    try:
        pred_lispress = render_compact(parse_lispress(pred_lispress))
    except:
        #pdb.set_trace()
        pred_lispress = "(Error)"
    gold_lispress = render_compact(parse_lispress(gold_lispress))
    pred = TurnPrediction(TurnId("test", 0), "", pred_lispress)
    true = TurnAnswer(TurnId("test", 0), "", gold_lispress, ProgramExecutionOracle(False, True))
    match, match_no_refer = evaluate_prediction_exact_match(pred, true)
    return match, match_no_refer


def read_nucleus_file(miso_pred_file):
    with open(miso_pred_file, "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    to_ret = []
    data_by_idx = defaultdict(list)
    data_by_src_str = defaultdict(list)
    for line in data:
        data_by_src_str[line['src_str']].append(line) 
        data_by_idx[line['line_idx']].append(line) 

    for src_str, lines in data_by_src_str.items():
        total_probs = [np.exp(np.sum(np.log(x['expression_probs']))) 
                                if x['expression_probs'] is not None else 0.0 
                                    for x in lines ]
        mean_probs = [np.mean(x['expression_probs']) 
                                if x['expression_probs'] is not None and np.sum(x['expression_probs']) > 0.0 
                                else 0.0 for x in lines ]
        min_probs = []
        for x in lines:
            if x['expression_probs'] is not None and len(x['expression_probs']) > 0:
                min_probs.append(np.min(x['expression_probs']))
            else:
                min_probs.append(0.0)

        combo_lines = zip(lines, min_probs, mean_probs, total_probs)
        sorted_combo_lines = sorted(combo_lines, key=lambda x: x[2], reverse=True)

        data_by_src_str[src_str] = sorted_combo_lines
        idx = lines[0]['line_idx']
        data_by_idx[idx] = sorted_combo_lines
    return data_by_src_str, data_by_idx

def read_gold_file(file):
    with open(file) as f:
        if file.endswith(".tgt"):
            to_ret = [render_compact(parse_lispress(line)) for line in f.readlines()]
        else:
            to_ret = [re.sub("__StartOfProgram", "", x).strip() for x in f.readlines()]
    return to_ret 

def read_benchclamp_file(path):
    with open(path) as f1:
        data = [json.loads(x) for x in f1]
    return data

def get_probs_and_accs_benchclamp(bclamp_data):
    min_probs, mean_probs, accs = [], [], []
    for line in bclamp_data: 
        is_correct = line['metrics']['exact_match/top1'] == "correct"
        token_probs = np.exp(line['token_logprobs'][0])
        # print(token_probs)
        min_seq_prob = np.min(token_probs) 
        mean_seq_prob = np.mean(token_probs) 
        min_probs.append(min_seq_prob)
        mean_probs.append(mean_seq_prob)
        accs.append(is_correct)
    return min_probs, mean_probs, accs



def get_probs_and_accs(nucleus_file, gold_src_file, gold_tgt_file):
    printed = 0
    __, nucleus = read_nucleus_file(nucleus_file)
    all_gold_tgt = read_gold_file(gold_tgt_file)
    all_gold_src = read_gold_file(gold_src_file)
    # assert(len(nucleus) == len(gold_tgt))
    min_probs = []
    mean_probs = []
    accs = []
    for i,  (gold_src, gold_tgt) in enumerate(zip(all_gold_src, all_gold_tgt)):
        nuc = nucleus[str(i)]
        try:
            nuc_str = nuc[0][0]['tgt_str']
        except KeyError:
            continue
            
        nuc_str = render_compact(parse_lispress(nuc_str))
        gold_tgt = render_compact(parse_lispress(gold_tgt))
        gold_tgt = re.sub('" (\w+) "', '"\g<1>"', gold_tgt)
        nuc_str = re.sub('" (\w+) "', '"\g<1>"', nuc_str)
        # use the min prob, not the summed prob 
        min_probs.append(nuc[0][1])
        # TODO (elias): add total number of tokens to get mean prob 
        mean_probs.append(nuc[0][2])
        # if nuc_str != gold_tgt and printed < 10:
        #     print(nuc_str)
        #     print(gold_tgt)
        #     printed += 1
        # accs.append(nuc_str == gold_tgt)
        match, __ = single_exact_match(nuc_str, gold_tgt)
        accs.append(match)

    print(f'Mean acc: {np.mean(accs)}')
    print(f"len accs: {len(accs)}")
    return min_probs, mean_probs, accs