import pdb 
import json 
import pathlib
import argparse
from tqdm import tqdm 

from transformers import AutoTokenizer
import torchtext 
import torch

from calibration_sql_lm import read_data, get_and_tokenize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(sents, vocab):
    data = []
    for example in sents: 
        example.append("<eos>")
        toks = [vocab[token] for token in example]
        data.append(toks)

    max_len = max([len(x) for x in data])
    for i, example in enumerate(data):
        data[i] = example + [vocab["<pad>"]] * (max_len - len(example))

    data = torch.LongTensor(data).to(device)
    # batch it up
    dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    return dataloader

def get_ppl(dataloader, model, vocab, criterion):
    # validation and test 
    model.eval()
    num_examples = 0
    with torch.no_grad():
        # total_loss = 0
        all_losses = []
        for i, batch in tqdm(enumerate(dataloader)):
            # shift by 1, cutting off last token 
            out = model(batch[:, :-1])
            # shift left by 1 for computing loss 
            loss = criterion(out.view(-1, len(vocab)), batch[:, 1:].reshape(-1))
            # inner_ppl = torch.exp(loss)
            # all_ppls.append(inner_ppl)
            all_losses.append(loss)
            # total_loss += loss
            num_examples += batch.shape[0]
        # flatten the list of tensors into single tensor
        all_losses = torch.cat(all_losses)
        ppl = torch.exp(torch.mean(all_losses)).item()
        # ppl = torch.exp(total_loss/num_examples)
        print(f"Perplexity: {ppl}")
    return ppl
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=None, required=True)
    parser.add_argument("--dev_file", type=str, default=None, required=True)
    parser.add_argument("--test_dir", type=pathlib.Path, default=None, required=True)
    parser.add_argument("--use_target", action='store_true')
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--model_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=None)
    args = parser.parse_args()

    train_data = read_data(args.train_file)
    dev_data = read_data(args.dev_file)
    if args.use_target:
        key = "plan"
    else:
        key = "utterance"

    # get tokenized sentences 
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 
    train_sents = get_and_tokenize(train_data, key, tokenizer, 1) 
    dev_sents = get_and_tokenize(dev_data, key, tokenizer, 1) 

    all_tokens = [token for sent in train_sents for token in sent]
    vocab = torchtext.vocab.build_vocab_from_iterator(train_sents, min_freq=5) 
    vocab.insert_token('<unk>', 0)           
    vocab.insert_token('<eos>', 1)  
    vocab.insert_token('<pad>', 2)  
    vocab.set_default_index(vocab['<unk>'])   

    train_dataloader = get_dataloader(train_sents, vocab)
    dev_dataloader = get_dataloader(dev_sents, vocab)

    # define LSTM LM
    class LSTM_LM(torch.nn.Module):
        def __init__(self):
            super(LSTM_LM, self).__init__()
            self.embedding = torch.nn.Embedding(len(vocab), args.model_size)
            self.lstm = torch.nn.LSTM(args.model_size, args.model_size, num_layers=2, batch_first=True)
            self.linear = torch.nn.Linear(args.model_size, len(vocab))

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = self.linear(x)
            return x

    model = LSTM_LM().to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    no_red_criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"], reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    detokenizer = vocab.get_itos()
    # train  to convergence 
    best_dev_ppl = float("inf")
    n_epochs_since_improvement = 0
    for epoch in range(100):
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            out = model(batch[:, :-1])
            loss = criterion(out.view(-1, len(vocab)), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss.item())

        dev_ppl = get_ppl(dev_dataloader, model, vocab, no_red_criterion)
        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            n_epochs_since_improvement = 0
        else:
            n_epochs_since_improvement += 1
        if n_epochs_since_improvement > 5:
            break 
        # sanity check: generate some sentences
        model.eval()
        with torch.no_grad():
            if not args.use_target:
                inputs = ["How many", "What is", "List"]
            else:
                inputs = ["SELECT COUNT", "SELECT", "SELECT * FROM"]
            outputs = []
            for inp in inputs:
                prefix = inp
                curr_seq = []
                # tokenize
                tok_inp = tokenizer.tokenize(inp)
                # convert to ids
                inp_ids = [vocab[token] for token in tok_inp]
                # convert to tensor
                inp_ids = torch.LongTensor(inp_ids).to(device)
                # add batch dimension
                inp_ids = inp_ids.unsqueeze(0)
                # generate
                curr_tok = inp_ids[:, -1]
                # curr_seq.append(curr_tok) # TODO: add seqs and decode
                for i in range(20):
                    out = model(inp_ids)
                    out = out[:, -1, :]
                    curr_tok = torch.argmax(out, dim=-1)
                    curr_seq.append(curr_tok)
                    inp_ids = torch.cat([inp_ids, curr_tok.unsqueeze(1)], dim=1)
                    if curr_tok == vocab["<eos>"]:
                        break

                curr_seq = torch.cat(curr_seq, dim=0)
                detokenized = [detokenizer[x] for x in curr_seq]
                # use tokenizer to get rid of special tokens
                detokenized = tokenizer.convert_tokens_to_string(detokenized)
                full_str = f"{prefix} {detokenized}"
                print(full_str)
                # outputs.append([detokenizer[x] for x in curr_seq])
                # pdb.set_trace()


    conf_and_ppl = []
    for test_file in args.test_dir.glob("*.jsonl"):
        bin_confidence = float(test_file.stem.split("_")[-1])
        print(f"file : {test_file}, confidence: {bin_confidence}")
        test_data = read_data(test_file)
        test_sents = get_and_tokenize(test_data, key, tokenizer, 1) 
        test_dataloader = get_dataloader(test_sents, vocab)
        test_ppl = get_ppl(test_dataloader, model, vocab, no_red_criterion)
        
        conf_and_ppl.append((bin_confidence, test_ppl))

    conf_and_ppl = sorted(conf_and_ppl, key=lambda x: x[0])
    tgt_str = "target" if args.use_target else "source"
    with open(f"{args.test_dir}/conf_and_ppl_{tgt_str}.json", "w") as f:
        json.dump(conf_and_ppl, f)
    for conf, ppl in conf_and_ppl:
        print(f"Confidence: {conf}, Average PPL: {ppl}")
