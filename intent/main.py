import argparse
import pathlib 
import json 
import pdb 
import random
from collections import defaultdict
path_to_min_pair_utils = pathlib.Path(__file__).resolve().parent.parent.joinpath("minimal_pair_utils")
import sys
sys.path.insert(0, str(path_to_min_pair_utils))
from construct import sort_train_by_min_pair

from tqdm import tqdm 
import torch
from torch import optim 
import numpy as np 
from datasets import load_dataset 

from model import Classifier
from data import batchify, batchify_min_pair, batchify_double_in_batch, batchify_double_in_data, split_by_intent, random_split 
from dro_loss import GroupDROLoss
def get_accuracy(pred, true, intent_of_interest = None):
    pred_classes = torch.argmax(pred, dim=1).detach().cpu()
    true = true.detach().cpu() 
    correct = torch.sum(pred_classes == true) 
    accuracy = float(correct) / true.shape[0]

    if intent_of_interest is not None: 
        true_idxs = true == intent_of_interest
        pred_of_interest = pred_classes[true_idxs]
        true_of_interest = true[true_idxs]
        intent_correct = torch.sum(pred_of_interest == true_of_interest) 
        if true_of_interest.shape[0] == 0:
            intent_accuracy = 0.0
        else:
            intent_accuracy = float(intent_correct) / true_of_interest.shape[0]
    else:
        intent_accuracy = 0.0

    return accuracy, intent_accuracy

def train_epoch(model, train_data, loss_fxn, optimizer, intent_of_interest):
    all_loss = []
    all_accs = []
    intent_accs = []
    model.train() 
    for batch in tqdm(train_data):
        optimizer.zero_grad() 
        pred_classes = model(batch)
        true_classes = batch['label']
        acc, intent_acc = get_accuracy(pred_classes, true_classes, intent_of_interest)
        all_accs.append(acc)
        intent_accs.append(intent_acc)
        loss = loss_fxn(pred_classes, true_classes)
        loss.backward()
        optimizer.step()
        all_loss.append(loss.item())
    return np.mean(all_loss), np.mean(all_accs), np.mean(intent_accs)

def eval_epoch(model, eval_data, loss_fxn, intent_of_interest = None):
    all_accs = []
    all_loss = []
    intent_accs = []
    model.eval() 
    with torch.no_grad():
        for batch in tqdm(eval_data):
            pred_classes = model(batch)
            true_classes = batch['label']
            acc, intent_acc = get_accuracy(pred_classes, true_classes, intent_of_interest)
            all_accs.append(acc)
            intent_accs.append(intent_acc)
            loss = loss_fxn(pred_classes, true_classes)
            all_loss.append(loss.item())
    return np.mean(all_loss), np.mean(all_accs), np.mean(intent_accs)

def generate_lookup_table(train_data, intent_of_interest):
    train_src, train_tgt, train_idx = [], [], []
    fxn_train_src, fxn_train_tgt, fxn_train_idx = [], [], []
    for i, example in enumerate(train_data):
        train_src.append(example['text'].strip().split(" "))
        train_tgt.append([example['label']])
        train_idx.append(i)
        if example['label'] == intent_of_interest:
            fxn_train_src.append(example['text'].strip().split(" "))
            fxn_train_tgt.append(example['label'])
            fxn_train_idx.append(i)
    print(f"Generating lookup table...")
    min_pair_lookup = defaultdict(list)
    for src, idx, tgt in tqdm(zip(fxn_train_src, fxn_train_idx, fxn_train_tgt)):
        min_pair_lookup[idx] = sort_train_by_min_pair(train_src, 
                                                      train_idx, 
                                                      train_tgt, 
                                                      src, 
                                                      tgt, 
                                                      args.intent_of_interest, 
                                                      num_mutants=1, 
                                                      names = [], 
                                                      mutation_types = ['identity'], 
                                                      do_sum = False,
                                                      fxn_frequencies=None,
                                                      anon_plan=False,
                                                      top_k=-1)
    return min_pair_lookup

def main(args):
    # set seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if checkpoint_dir.joinpath("best.th").exists():
        raise AssertionError(f"Checkpoint dir {checkpoint_dir} is not empty! Will not overwrite")

    checkpoint_dir.joinpath("data").mkdir(exist_ok=True, parents=True)

    # get data 
    dataset = load_dataset("nlu_evaluation_data")
    if args.split_type == "random": 
        train_data, dev_data, test_data = random_split(dataset, 0.7, 0.1, 0.2)
    else:
        train_data, dev_data, test_data = split_by_intent(dataset, 
                                                          args.intent_of_interest,
                                                          args.total_train,
                                                          args.total_interest,
                                                          out_path = checkpoint_dir.joinpath("data"))

    if args.batch_min_pairs:
        lookup_table = generate_lookup_table(train_data, args.intent_of_interest)
        train_batches = batchify_min_pair(train_data, args.batch_size, args.bert_name, device, args.intent_of_interest, lookup_table)
        dev_batches, test_batches = [batchify(x, args.batch_size, args.bert_name, device) for x in [dev_data, test_data]]
    elif args.double_in_batch:
        train_batches = batchify_double_in_batch(train_data, args.batch_size, args.bert_name, device, args.intent_of_interest)
        dev_batches, test_batches = [batchify(x, args.batch_size, args.bert_name, device) for x in [dev_data, test_data]]
    elif args.double_in_data:
        train_batches = batchify_double_in_data(train_data, args.batch_size, args.bert_name, device, args.intent_of_interest)
        dev_batches, test_batches = [batchify(x, args.batch_size, args.bert_name, device) for x in [dev_data, test_data]]
    else:
        train_batches, dev_batches, test_batches = [batchify(x, args.batch_size, args.bert_name, device) for x in [train_data, dev_data, test_data]]

    # make model and optimizer
    model = Classifier(args.bert_name)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # train
    if not args.do_dro:
        loss_fxn = torch.nn.CrossEntropyLoss()
    else:
        loss_fxn = GroupDROLoss()
    best_epoch = 0
    best_acc = -1
    epochs_without_change = 0
    for e in range(args.epochs):
        print(f"training epoch {e}")
        train_loss, train_acc, interest_train_acc = train_epoch(model, train_batches, loss_fxn, optimizer, args.intent_of_interest)
        dev_loss, dev_acc, interest_dev_acc = eval_epoch(model, dev_batches, loss_fxn, args.intent_of_interest) 
        print(f"TRAIN loss/acc: {train_loss}, {train_acc:0.1%}, DEV loss/acc: {dev_loss}, {dev_acc:0.1%}")
        if args.intent_of_interest is not None: 
            print(f"TRAIN {args.intent_of_interest} acc: {interest_train_acc:0.1%}, DEV acc: {interest_dev_acc:0.1%}")

        with open(checkpoint_dir.joinpath(f"train_metrics_{e}.json"), "w") as f1:
            data_to_write = {"epoch": e, "acc": train_acc, f"{args.intent_of_interest}_acc": interest_train_acc, "loss": train_loss}
            json.dump(data_to_write, f1)
        with open(checkpoint_dir.joinpath(f"dev_metrics_{e}.json"), "w") as f1:
            data_to_write = {"epoch": e, "acc": dev_acc, f"{args.intent_of_interest}_acc": interest_dev_acc, "loss": dev_loss}
            json.dump(data_to_write, f1)

        if dev_acc > best_acc:
            best_acc = dev_acc 
            print(f"new best at epoch {e}: {dev_acc:0.1%}")
            with open(checkpoint_dir.joinpath("best_dev_metrics.json"), "w") as f1:
                data_to_write = {"best_epoch": e, "best_acc": dev_acc, f"best_{args.intent_of_interest}_acc": interest_dev_acc}
                json.dump(data_to_write, f1)
            torch.save(model.state_dict(), checkpoint_dir.joinpath("best.th"))
            epochs_without_change = 0

        if epochs_without_change > args.patience:
            print(f"Ran out of patience!")
            break 

    print(f"evaluating model...")
    print(f"loading best weights from {checkpoint_dir.joinpath('best.th')}")
    model.load_state_dict(torch.load(checkpoint_dir.joinpath("best.th")))
    test_loss, test_acc, interest_test_acc = eval_epoch(model, test_batches, loss_fxn, 
                                                        intent_of_interest = args.intent_of_interest) 

    with open(checkpoint_dir.joinpath("test_metrics.json"), "w") as f1:
        data_to_write = {"epoch": e, "acc": test_acc, 
                         f"{args.intent_of_interest}_acc": interest_test_acc, 
                         "loss": test_loss}
        json.dump(data_to_write, f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data 
    parser.add_argument("--split-type", default="random", choices=["random", "interest"], 
                        required=True, help="type of datasplit to train on")
    parser.add_argument("--intent-of-interest", default=None, type=int, help="intent to look at") 
    parser.add_argument("--total-train", type=int, default=None, help = "total num training examples") 
    parser.add_argument("--total-interest", type=int, default=None, help = "total num intent of interest examples") 
    parser.add_argument("--batch-min-pairs", action="store_true", help="flag to set if you want to train with minimal pair batching")
    parser.add_argument("--double-in-batch", action="store_true", help="flag to set if you want to double examples of interest in batch")
    parser.add_argument("--double-in-data", action="store_true", help="flag to set if you want to double examples of interest in data")

    # Model/Training
    parser.add_argument("--bert-name", default="bert-base-cased", required=True, help="bert pretrained model to use")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-5, help="learn rate to use") 
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="path to save models and logs")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10, help="how many epochs to wait for without improvement before early stopping")
    parser.add_argument("--do-dro", action="store_true", help="flag to do group DRO over intents")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args() 

    main(args)