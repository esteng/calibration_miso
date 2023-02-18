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
        total_loss = 0
        for i, batch in tqdm(enumerate(dataloader)):
            out = model(batch[:, :-1])
            loss = criterion(out.view(-1, len(vocab)), batch[:, 1:].reshape(-1))
            total_loss += loss
            num_examples += batch.shape[0]
        ppl = torch.exp(total_loss/num_examples)
        print(f"Perplexity: {ppl}")
    return ppl
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=None, required=True)
    parser.add_argument("--dev_file", type=str, default=None, required=True)
    parser.add_argument("--test_dir", type=pathlib.Path, default=None, required=True)
    parser.add_argument("--use_target", action='store_true')
    parser.add_argument("--tokenizer", type=str, default="gpt2")
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
    vocab = torchtext.vocab.build_vocab_from_iterator(all_tokens, min_freq=3) 
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
            self.embedding = torch.nn.Embedding(len(vocab), 256)
            self.lstm = torch.nn.LSTM(256, 256, num_layers=2, batch_first=True)
            self.linear = torch.nn.Linear(256, len(vocab))

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = self.linear(x)
            return x

    model = LSTM_LM().to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train 
    for epoch in range(10):
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            out = model(batch[:, :-1])
            loss = criterion(out.view(-1, len(vocab)), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss.item())

        dev_ppl = get_ppl(dev_dataloader, model, vocab, criterion)
        # sanity check: generate some sentences
        model.eval()
        with torch.no_grad():
            inputs = ["How many", "What is", "List"]
            for inp in inputs:
                # tokenize
                inp = tokenizer.tokenize(inp)
                # convert to ids
                inp = [vocab[token] for token in inp]
                # convert to tensor
                inp = torch.LongTensor(inp).to(device)
                # add batch dimension
                inp = inp.unsqueeze(0)
                # generate
                curr_tok = inp[:, -1]
                seq # TODO: add seqs and decode
                for i in range(20):
                    out = model(inp)
                    out = out[:, -1, :]
                    curr_tok = torch.argmax(out, dim=-1)
                    inp = torch.cat([inp, curr_tok.unsqueeze(1)], dim=1)


    conf_and_ppl = []
    for test_file in args.test_dir.glob("*.jsonl"):
        bin_confidence = float(test_file.stem.split("_")[-1])
        print(f"file : {test_file}, confidence: {bin_confidence}")
        test_data = read_data(test_file)
        test_sents = get_and_tokenize(test_data, key, tokenizer, 1) 
        test_dataloader = get_dataloader(test_sents, vocab)
        test_ppl = get_ppl(test_dataloader, model, vocab, criterion)
        
        conf_and_ppl.append((bin_confidence, test_ppl))

    conf_and_ppl = sorted(conf_and_ppl, key=lambda x: x[0])
    for conf, ppl in conf_and_ppl:
        print(f"Confidence: {conf}, Average PPL: {ppl}")
