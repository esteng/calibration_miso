import pdb 
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

def split_by_intent(dataset, intent_of_interest, n_data, n_intent): 
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

