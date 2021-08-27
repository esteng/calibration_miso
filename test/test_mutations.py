import pytest 
import os 
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 
from miso.data.batching.mutations import DeleteMutation, SwapMutation


def test_delete_mutation_basic():
    src_text = "Make a meeting with Jim and my manager".split(" ")
    mutation = DeleteMutation()
    for i in range(10):
        mutant = mutation(src_text)
        print(" ".join(mutant))
    src_text = "Cancel the meeting with Brenda and her boss today .".split(" ")
    mutation = DeleteMutation()
    for i in range(10):
        mutant = mutation(src_text)
        print(" ".join(mutant))

    src_text = "Can you find my appointment in January with Sal and his boss ?".split(" ")
    mutation = DeleteMutation()
    for i in range(10):
        mutant = mutation(src_text)
        print(" ".join(mutant))

    src_text = "Who does he report to ?".split(" ")
    mutation = DeleteMutation()
    for i in range(10):
        mutant = mutation(src_text)
        print(" ".join(mutant))

    assert(1==2)

def test_swap_mutation_basic():
    train_path = "test/data/smcalflow.agent.data/train.src_tok"
    train_lines = [x.strip().split(" ") for x in open(train_path).readlines()]

    for i, tl in enumerate(train_lines):
        start_idxs = [j for j, w in enumerate(tl) if w == "__User"]
        # start at __User and remove __Start
        tl = tl[start_idxs[-1]+1:-1]
        train_lines[i] = tl

    src_texts = ["Make a meeting with Jim and my manager".split(" "),
                "Cancel the meeting with Brenda and her boss today .".split(" "),
                "Can you find my appointment in January with Sal and his boss ?".split(" "),
                "Who does he report to ?".split(" ")]
    for src_text in src_texts:
        mutation = SwapMutation.from_train(src_text, train_lines)
        for i in range(10):
            mutant = mutation(src_text)
            print(" ".join(mutant))
    assert(1==2)