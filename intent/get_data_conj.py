import json
import numpy as np
import pathlib
import pdb 
np.random.seed(12) 

def filter_data(path, diverse=True): 
    with open(path) as f1:
        data = json.load(f1) 

    fifty_examples = []
    play_examples = []
    for ex in data: 
        if "play" in ex['text'].split(" ") and ex['label'] != 50:
            play_examples.append(ex)
        if ex['label'] == 50:
            fifty_examples.append(ex) 

    if not diverse:    
        other_play_examples = [x for x in play_examples if x['label'] == 48]
        other_nonplay_examples = [x for x in data if x['label'] == 48 and 'play' not in ex['text'].split(" ")]
    else:
        other_play_examples = [x for x in play_examples if x['label'] != 50]
        other_nonplay_examples = [x for x in data if x['label'] != 50 and 'play' not in ex['text'].split(" ")]

    radio_examples = [x for x in fifty_examples if "play" in x['text'].split(" ")]

    return other_play_examples, other_nonplay_examples, radio_examples

def pick_data(trigger_ex, nontrigger_ex, intent_ex, perc, ratio):
    # create a dataset with ratio of itnent to nonintent and a percentage perc of nonintent with play
    #num_intent = len(intent_ex)
    num_intent = 30
    #num_other = perc * len(trigger_ex) + (1 - perc) * len(nontrigger_ex) 
   
    #num_intent = min(ratio * num_other, len(intent_ex)) 
    #num_intent = min(num_intent, ratio * (len(trigger_ex) + len(nontrigger_ex)))

    num_other =  num_intent/ratio
    num_trigger = int(perc * num_other)
    num_nontrigger = int(num_other) - num_trigger 

    # print(num_intent, len(intent_ex))
    # assert(num_intent <= len(intent_ex))
    #num_other = int(num_other)
    # num_intent = int(num_intent) 
    # num_trigger = int(perc * len(trigger_ex))
    # num_nontrigger = int((1- perc) * len(nontrigger_ex))
    
    intent_ex_sample = np.random.choice(intent_ex, num_intent).tolist()
    trigger_ex_sample = np.random.choice(trigger_ex, num_trigger, replace=True).tolist()
    nontrigger_ex_sample = np.random.choice(nontrigger_ex, num_nontrigger, replace=True).tolist()

    train_data = intent_ex_sample+trigger_ex_sample+nontrigger_ex_sample
    np.random.shuffle(train_data) 
    return train_data, len(intent_ex_sample), len(trigger_ex_sample), len(nontrigger_ex_sample)
    
if __name__ == "__main__": 
    music_play_examples, music_nonplay_examples, radio_examples = filter_data("data/nlu_eval_data/train.json", diverse=True)
    dev_music_play_examples, dev_music_nonplay_examples, dev_radio_examples = filter_data("data/nlu_eval_data/dev.json", diverse=True)
    test_music_play_examples, test_music_nonplay_examples, test_radio_examples = filter_data("data/nlu_eval_data/test.json", diverse=True)

    dev_examples = dev_music_nonplay_examples + dev_music_play_examples + dev_radio_examples
    test_examples = test_music_nonplay_examples + test_music_play_examples + test_radio_examples


    with open('data/conj_data_diverse/dev.json', 'w') as dev_f, open('data/conj_data_diverse/test.json','w') as test_f: 
        json.dump(dev_examples, dev_f)
        json.dump(test_examples, test_f)

    #all_perc_play_music = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_perc_play_music = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_ratios = [1, 0.5, 0.25, 0.1]

    for perc in all_perc_play_music:
        for ratio in all_ratios: 
            training_data, len_intent, len_trigger, len_nontrigger = pick_data(music_play_examples, music_nonplay_examples, radio_examples, perc, ratio)

            emp_ratio = len_intent / (len_trigger + len_nontrigger)
            try:
                emp_perc = len_trigger / (len_nontrigger + len_trigger) 
            except ZeroDivisionError:
                continue

            if emp_perc > 1:
                pdb.set_trace() 

            out_path = pathlib.Path(f"data/conj_data_diverse/{emp_ratio:.2f}_{emp_perc:.2f}/")
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path.joinpath("train.json")
            
            with open(out_path, "w") as f1:
                json.dump(training_data, f1)
            