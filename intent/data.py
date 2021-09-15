import pdb 
import pathlib
import numpy as np
import torch 
from transformers import AutoTokenizer
from datasets import load_dataset 
np.random.seed(12) 

def random_split(dataset, p_train, p_dev, p_test): 
    dataset = dataset["train"]
    num_examples = len(dataset) 
    idxs = [i for i in range(num_examples)]
    np.random.shuffle(idxs)
    train_end = int(p_train * num_examples) 
    dev_start = train_end 
    dev_end = train_end + int( p_dev * num_examples) 
    test_start = dev_end 

    print(f"train end: {train_end}, dev end: {dev_end}") 
    train_idxs = idxs[0: train_end]
    dev_idxs = idxs[dev_start:dev_end]
    test_idxs = idxs[test_start:]
    train_data = [dataset[i] for i in train_idxs]
    dev_data = [dataset[i] for i in dev_idxs]
    test_data = [dataset[i] for i in test_idxs]
    return train_data, dev_data, test_data 

def split_by_intent(dataset, intent_of_interest, n_data, n_intent, out_path = None): 
    dataset = dataset['train']
    # split into interest and non-interest 
    of_interest = [i for i, x in enumerate(dataset) if x['label'] == intent_of_interest]
    not_interest = [i for i in range(len(dataset)) if i not in of_interest]
    np.random.shuffle(of_interest)
    np.random.shuffle(not_interest)

    train_idxs = not_interest[0:n_data - n_intent] + of_interest[0:n_intent]
    np.random.shuffle(train_idxs)

    remaining = [i for i in range(len(dataset)) if i not in train_idxs]
    np.random.shuffle(remaining)
    dev_idxs = remaining[0:int(len(remaining)/2)]
    test_idxs = remaining[int(len(remaining)/2):]
    train_data = [dataset[i] for i in train_idxs]
    dev_data = [dataset[i] for i in dev_idxs]
    test_data = [dataset[i] for i in test_idxs]

    n_interest_in_dev = np.sum([1 for x in dev_data if x['label'] == intent_of_interest])
    n_interest_in_test = np.sum([1 for x in test_data if x['label'] == intent_of_interest])
    print(f"There are {n_interest_in_dev} instances of {intent_of_interest} in dev and {n_interest_in_test} in test") 

    if out_path is not None:
        out_path = pathlib.Path(out_path)
        with open(out_path.joinpath("train.src_tok"), "w") as src_f, open(out_path.joinpath("train.tgt"), "w") as tgt_f:
            for i, datapoint in enumerate(train_data):
                src_f.write(datapoint['text'].strip() + "\n")
                tgt_f.write(str(datapoint['label']).strip() + "\n")

    return train_data, dev_data, test_data 

def batchify(data, batch_size, bert_model, device):
    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": []}
    curr_batch_as_text = {"input": [], "label": []}
    for chunk_start in range(0, len(data), batch_size):
        for example in data[chunk_start: chunk_start + batch_size]:
            label = example['label']
            text = example['text']
            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        curr_batch = {"input": [], "label": []}
        curr_batch_as_text = {"input": [], "label": []}
    return batches 


def batchify_min_pair(data, batch_size, bert_model, device, intent_of_interest, lookup_table):
    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": []}
    curr_batch_as_text = {"input": [], "label": []}
    # add indices 
    data_lookup_dict = {}
    for i, datapoint in enumerate(data):
        datapoint['idx'] = i
        data[i] = datapoint
        data_lookup_dict[i] = (datapoint['text'].strip(), datapoint['label'])

    done = set()
    increased_batch_size = False
    for i, example in enumerate(data):
        # if current batch is right size, append and do next 
        if increased_batch_size or (len(curr_batch_as_text['input']) > 0 and len(curr_batch_as_text['input'])  % batch_size == 0):
            all_text = curr_batch_as_text['input'] 
            tokenized = tokenizer(all_text,  padding=True)
            ids = tokenized['input_ids']
            curr_batch['input'] = ids
            curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
            curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

            batches.append(curr_batch)
            curr_batch = {"input": [], "label": []}
            curr_batch_as_text = {"input": [], "label": []}
            # if batch size was increased in earlier iteration, decrease again 
            if increased_batch_size:
                batch_size -= 1
                increased_batch_size = False 

        # don't use same example twice
        if example['idx'] in done:
            continue

        # get an example 
        label = example['label']
        text = example['text']
        # if it's an example of interest, get its min pair
        if label == intent_of_interest:
            top_pair_idxs = lookup_table[i]
            i = 0
            while  top_pair_idxs[i] in done:
                i+=1
            tp_idx = top_pair_idxs[i]

        # edge case: the example of interest is last in the batch, 
        # and adding one more will put over batch size, then increase 
        # batch size by 1 temporarily 
        if (len(curr_batch_as_text['input']) + 1) % batch_size == 0:
            batch_size += 1
            increased_batch_size = True

        # Add the example 
        curr_batch_as_text['input'].append(text)
        curr_batch['label'].append(label)
        done.add(example['idx'])
        # if appropriate, add min pair 
        if label == intent_of_interest:
            top_pair = data_lookup_dict[tp_idx]
            print(f"batching {text}:{label} with {top_pair[0]}:{top_pair[1]}")
            curr_batch_as_text['input'].append(top_pair[0])
            curr_batch['label'].append(top_pair[1])
            done.add(tp_idx)

    # add last batch
    if len(curr_batch_as_text['input']) > 0: 
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        
    return batches 

def batchify_double_in_batch(data, batch_size, bert_model, device, intent_of_interest):
    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": []}
    curr_batch_as_text = {"input": [], "label": []}

    increased_batch_size = False
    for i, example in enumerate(data):
        # if current batch is right size, append and do next 
        if increased_batch_size or (len(curr_batch_as_text['input']) > 0 and len(curr_batch_as_text['input'])  % batch_size == 0):
            all_text = curr_batch_as_text['input'] 
            tokenized = tokenizer(all_text,  padding=True)
            ids = tokenized['input_ids']
            curr_batch['input'] = ids
            curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
            curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

            batches.append(curr_batch)
            curr_batch = {"input": [], "label": []}
            curr_batch_as_text = {"input": [], "label": []}
            # if batch size was increased in earlier iteration, decrease again 
            if increased_batch_size:
                batch_size -= 1
                increased_batch_size = False 

        # get an example 
        label = example['label']
        text = example['text']
        curr_batch_as_text['input'].append(text)
        curr_batch['label'].append(label)

        # if it's an example of interest, double it 
        if label == intent_of_interest:
            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)

        # edge case: the example of interest is last in the batch, 
        # and adding one more will put over batch size, then increase 
        # batch size by 1 temporarily 
        if (len(curr_batch_as_text['input']) + 1) % batch_size == 0:
            batch_size += 1
            increased_batch_size = True

    # add last batch
    if len(curr_batch_as_text['input']) > 0: 
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        
    return batches 


def batchify_double_in_data(data, batch_size, bert_model, device, intent_of_interest):
    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": []}
    curr_batch_as_text = {"input": [], "label": []}

    # double in data
    new_data =[]
    for i, example in enumerate(data):
        new_data.append(example)
        if example['label'] == intent_of_interest:
            new_data.append(example)

    # shuffle 
    np.random.shuffle(new_data)

    for chunk_start in range(0, len(new_data), batch_size):
        for example in data[chunk_start: chunk_start + batch_size]:
            label = example['label']
            text = example['text']
            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        curr_batch = {"input": [], "label": []}
        curr_batch_as_text = {"input": [], "label": []}
    return batches 