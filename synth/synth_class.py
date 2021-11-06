import pathlib
from argparse import ArgumentParser
import argparse
import json 
from tqdm import tqdm 
import pdb 

import torch
import numpy as np

class SynthModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers): 
        super(SynthModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers 

        self.embedder = torch.nn.Embedding(29, input_dim, padding_idx=28)
        self.encoder = torch.nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, dropout=0.2)
        self.classifier = torch.nn.Linear(hidden_dim, 2)

        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.id_to_token = {i: c for i, c in enumerate(alphabet)}
        self.token_to_id = {c: i for i, c in enumerate(alphabet)}
        self.id_to_token[28] = 'P'
        self.token_to_id[28] = 'P'

    def prepare_batch(self, data): 
        inputs = []
        outputs = []
        lens = []
        pos = [0,1]
        neg = [1,0]
        max_len = max([len(x[0]) for x in data])
        for inp, outp in data:
            inp_as_ids = [self.token_to_id[x] for x in inp]
            lens.append(len(inp_as_ids))
            inp_as_ids += [28 for i in range(max_len - len(inp_as_ids))]
            inputs.append(inp_as_ids)
            if outp == 0:
                outputs.append(neg)
            else:
                outputs.append(pos)
        return torch.LongTensor(inputs), torch.LongTensor(outputs), torch.LongTensor(lens) 

    def forward(self, batch_inputs, batch_lens): 
        embedded = self.embedder(batch_inputs)
        encoded = self.encoder(embedded)[0] 
        final = encoded[:, batch_lens, :]
        output = self.classifier(final)
        return output 

class SynthTrainer:
    def __init__(self, model, checkpoint_dir, path, n_epochs, batch_size, lr, patience = 40): 
        self.checkpoint_dir = pathlib.Path(checkpoint_dir) 
        self.checkpoint_dir.mkdir()

        self.n_epochs = n_epochs
        self.batch_size = batch_size 
        self.model = model 
        
        path = pathlib.Path(path) 
        with open(path.joinpath("train.json")) as train_f, \
            open(path.joinpath("dev.json")) as dev_f, \
            open(path.joinpath("test.json")) as test_f: 
            self.train_data = json.load(train_f)
            self.dev_data = json.load(dev_f)
            self.test_data = json.load(test_f)

        self.train_batches = self.get_batches(self.train_data)
        self.dev_batches = self.get_batches(self.dev_data)
        self.test_batches = self.get_batches(self.test_data)


        self.loss_fxn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr = lr)

        self.patience = patience 

    def get_batches(self, data): 
        batches = []
        curr_batch = []
        for i in range(len(data)): 
            curr_batch.append(data[i])
            if (i+1) % self.batch_size == 0: 
                batches.append(curr_batch)
                curr_batch = []
        if len(curr_batch) > 0:
            batches.append(curr_batch)
        return batches 

    def get_acc(self, pred, true): 
        pred = torch.Tensor(pred)
        true = torch.Tensor(true)
        max_pred_idxs = torch.argmax(pred, dim=1)
        max_true_idxs = torch.argmax(true, dim=1)

        correct = torch.sum(max_pred_idxs == max_true_idxs)
        total = len(pred)

        zero_idxs = true == 0
        one_idxs = true == 1 
        pred_zero = max_pred_idxs[zero_idxs]
        true_zero = max_true_idxs[zero_idxs]
        correct_zero = torch.sum(pred_zero == true_zero)
        total_zero = len(pred_zero)
        pred_one = max_pred_idxs[one_idxs]
        true_one = max_true_idxs[one_idxs]
        correct_one = torch.sum(pred_one == true_one)
        total_one = len(pred_one)
        return correct/total, correct_zero/total_zero, correct_one/total_one

    def train(self): 
        best_epoch = -1 
        best_acc = -1.0   
        for i in tqdm(range(self.n_epochs)): 
            self.model.train()
            for b in self.train_batches:
                inputs, outputs, lens = self.model.prepare_batch(b) 
                pred = self.model(inputs, lens) 
                self.optimizer.zero_grad()
                loss = self.loss_fxn(pred, outputs)
                loss.backward()
                self.optimizer.step()

            self.model.eval()  
            flat_preds = []
            flat_trues = []
            for d in self.dev_batches:
                inputs, outputs, lens = self.model.prepare_batch(d) 
                flat_trues += [x for x in outputs]
                with torch.no_grad():
                    pred = self.model(inputs, lens)
                    flat_preds += [x for x in pred]
            total_acc, zero_acc, one_acc = self.get_acc(flat_preds, flat_trues)

            print(f"Epoch {i}, total acc {total_acc * 100:.2f}, zero acc: {zero_acc * 100:.2f}, one acc: {one_acc * 100 :.2f} ")

            with open(self.checkpoint_dir.joinpath(f"dev_metrics_{i}.json"), "w") as f1: 
                json.dump({"total": total_acc, "zero": zero_acc, "one": one_acc}, f1, indent=4)
            if total_acc > best_acc:
                print(f"New best at {i}: {total_acc * 100:.2f}")
                best_acc = total_acc
                best_epoch = i 
                torch.save(self.model.state_dict(), self.checkpoint_dir.joinpath("best.th"))
            if i - best_epoch > self.patience:
                print(f"Ran out of patience!")
                break

        self.test() 

    def test(self): 
        state_dict = torch.load(self.checkpoint_dir.joinpath("best.th"))
        self.model.load_(state_dict)

        self.model.eval() 
        with torch.no_grad():
            flat_preds, flat_trues = [], []
            for t in self.test_batches: 
                inputs, outputs, lens = self.model.prepare_batch(t)  
                flat_trues += [x for x in outputs]
                pred = self.model(inputs, lens)
                flat_preds += [x for x in pred]
        total_acc, zero_acc, one_acc = self.get_acc(flat_preds, flat_trues)
        print(f"TEST: total acc {total_acc * 100:.2f}, zero acc: {zero_acc * 100:.2f}, one acc: {one_acc * 100 :.2f} ")
        with open(self.checkpoint_dir.joinpath("test_metrics.json"), "w") as f1: 
            json.dump({"total": total_acc, "zero": zero_acc, "one": one_acc}, f1, indent=4)

    
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", default=12, type=int)
    args = parser.parse_args()
    np.random.seed(args.seed)
    # torch.set_seed(args.seed)

    model = SynthModel(16, 32, 1)
    trainer = SynthTrainer(model, args.checkpoint_dir, args.path, args.n_epochs, args.batch_size, args.lr)

    trainer.train() 