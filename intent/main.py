import argparse
import pathlib 
import json 
import pdb 
import random 

from tqdm import tqdm 
import torch
from torch import optim 
import numpy as np 
from datasets import load_dataset 

from model import Classifier
from data import batchify, split_by_intent, random_split 

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

def main(args):
    # set seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda:0")
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if checkpoint_dir.joinpath("best.th").exists():
        raise AssertionError(f"Checkpoint dir {checkpoint_dir} is not empty! Will not overwrite")

    # get data 
    dataset = load_dataset("nlu_evaluation_data")
    if args.split_type == "random": 
        train_data, dev_data, test_data = random_split(dataset, 0.7, 0.1, 0.2)
    else:
        train_data, dev_data, test_data = split_by_intent(dataset, 
                                                          args.intent_of_interest,
                                                          args.total_train,
                                                          args.total_interest)

    train_batches, dev_batches, test_batches = [batchify(x, args.batch_size, args.bert_name, device) for x in [train_data, dev_data, test_data]]

    # make model and optimizer
    model = Classifier(args.bert_name)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # train
    loss_fxn = torch.nn.CrossEntropyLoss()
    best_epoch = 0
    best_acc = -1
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

    # Model/Training
    parser.add_argument("--bert-name", default="bert-base-cased", required=True, help="bert pretrained model to use")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-5, help="learn rate to use") 
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="path to save models and logs")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args() 

    main(args)