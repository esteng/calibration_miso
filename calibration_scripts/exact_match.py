import argparse
from re import A
from ssl import match_hostname
from dataflow.core.lispress import parse_lispress, render_compact
from dataflow.leaderboard.evaluate import evaluate_prediction_exact_match 
from dataflow.core.turn_prediction import TurnPrediction, TurnAnswer
from dataflow.core.dialogue import TurnId, ProgramExecutionOracle
import pdb 

def single_exact_match(pred_lispress, gold_lispress):
    pred_lispress = render_compact(parse_lispress(pred_lispress))
    gold_lispress = render_compact(parse_lispress(gold_lispress))
    pred = TurnPrediction(TurnId("test", 0), "", pred_lispress)
    true = TurnAnswer(TurnId("test", 0), "", gold_lispress, ProgramExecutionOracle(False, True))
    match, match_no_refer = evaluate_prediction_exact_match(pred, true)
    return match, match_no_refer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    args = parser.parse_args()

    with open(args.pred, "r") as f:
        pred_lines = f.readlines()

    with open(args.gold, "r") as f:
        gold_lines = f.readlines()

    assert(len(pred_lines) == len(gold_lines))
    total = 0
    correct = 0
    correct_no_refer = 0
    for pred, gold in zip(pred_lines, gold_lines): 
        match, match_no_refer = single_exact_match(pred, gold) 
        if match:
            correct += 1
        if match_no_refer:
            correct_no_refer += 1
        total += 1

    print(f"Accuracy: {correct/total * 100:.2f}") 
    print(f"Accuracy (no refer): {correct_no_refer/total * 100:.2f}") 