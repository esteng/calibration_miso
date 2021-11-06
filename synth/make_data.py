import argparse
from collections import defaultdict
from typing import DefaultDict
import numpy as np 
import pathlib 
import json 
from collections import defaultdict

np.random.seed(12) 

def generate_examples(n_examples, n_intent, p_intent_given_trigger): 
    # inputs should be unambiguous as a whole 
    # but should have triggers that are shared with some prob 
    # real example: 
        # play station 104.3 FM 
        # who plays baseball
    # synthetic example 
        # a b c d
        # e a f

    a_alpha = [x for x in 'acdefghijklm']
    b_alpha = [x for x in 'nopqrstuvwxy']

    non_examples = []
    yes_examples = []
    for i in range(n_examples - n_intent): 
        l = np.random.choice([2,3,4,5,6,7,8,9,10])
        input_non = [np.random.choice(a_alpha) for i in range(l)]
        prob_of_trigger = 1 - p_intent_given_trigger
        #add_trigger = np.random.choice([True, False], p=[prob_of_trigger, 1-prob_of_trigger])
        #if add_trigger:
        #    idx = np.random.choice(len(input_non)-1)
        #    input_non[idx] = 'b'
        non_examples.append((input_non, 0))

    for i in range(n_intent):
        l = np.random.choice([2,3,4,5,6,7,8,9,10])
        input_yes = [np.random.choice(b_alpha) for i in range(l)]
        prob_of_trigger = p_intent_given_trigger
        #add_trigger = np.random.choice([True, False], p=[prob_of_trigger, 1-prob_of_trigger])
        #if add_trigger:
        #    idx = np.random.choice(len(input_yes)-1)
        #    input_yes[idx] = 'b'
        yes_examples.append((input_yes, 1))

    # enforce prob 
    perc_b_and_1 = p_intent_given_trigger
    #num_1 = n_intent
    num_intent_with_b = int(perc_b_and_1 * n_intent)
    print(num_intent_with_b)
    num_non_intent_with_b = n_intent - num_intent_with_b
    print(num_non_intent_with_b)
    # perc_b_and_1  = p1 * n_intent + (1-p1) * 
    for i, (input, output) in enumerate(yes_examples[0:num_intent_with_b]):
        idx = np.random.choice(len(input)-1)
        input[idx] = 'b'
        yes_examples[i] = (input, output) 
    for i, (input, output) in enumerate(non_examples[0:num_non_intent_with_b]):
        idx = np.random.choice(len(input)-1)
        input[idx] = 'b'
        non_examples[i] = (input, output) 

    examples =  yes_examples + non_examples
    np.random.shuffle(examples)
    return examples 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--n-examples", default=10000, type=int, help="number of examples to generate")
    parser.add_argument("--n-intent", type=int, default=100,  help="number of examples for the intent")
    parser.add_argument("--p-intent-given-trigger", type=float, default=0.9)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    examples = generate_examples(args.n_examples, args.n_intent, args.p_intent_given_trigger)

    n_train = int(0.7 * len(examples))
    n_dev = int(0.1* len(examples))
    train_examples = examples[0:n_train]
    dev_examples = examples[n_train:n_train + n_dev]
    test_examples = examples[n_train + n_dev:]

    output_path = pathlib.Path(args.out_dir).joinpath(str(args.p_intent_given_trigger))
    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path.joinpath("train.json"), "w") as train_f, \
        open(output_path.joinpath("dev.json"), "w") as dev_f, \
        open(output_path.joinpath("test.json"), "w") as test_f: 

        json.dump(train_examples, train_f)
        json.dump(dev_examples, dev_f)
        json.dump(test_examples, test_f)


    # check prob intent given b
    count_intent_given_trigger = defaultdict(lambda: defaultdict(int))
    count_trigger = defaultdict(int)
    count_intent = defaultdict(int)
    for input, output in train_examples: 
        for char in input:
            count_intent_given_trigger[char][output] += 1
            count_trigger[char] += 1
        count_intent[output] += 1

    prob_intent_given_trigger = defaultdict(lambda: defaultdict(float))
    for char, dist in count_intent_given_trigger.items():
        for intent, count in dist.items():
            prob_intent_given_trigger[char][intent] = count / count_trigger[char]

    print(f"prob class 0 given b: {prob_intent_given_trigger['b'][0]}") 
    print(f"prob class 1 given b: {prob_intent_given_trigger['b'][1]}") 

        