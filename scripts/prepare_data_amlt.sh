#!/bin/bash

for dataset in "smcalflow"; do
    processed_text_data_dir="~/resources/data/smcalflow.agent.data"
    mkdir -p "${processed_text_data_dir}"
    dataflow_dialogues_dir="~/resources/data/smcalflow.agent.data" 
    #for subset in  "train" "valid" "dev_valid" "test_valid"; do
    for subset in "test"; do 
        python -m dataflow.onmt_helpers.create_onmt_text_data \
            --dialogues_jsonl ${dataflow_dialogues_dir}/${subset}.dataflow_dialogues.jsonl \
            --num_context_turns 1 \
	    --include_agent_utterance \
            --onmt_text_data_outbase ${processed_text_data_dir}/${subset}
            #--include_program \
            #--include_described_entities \
    done
done

python scripts/split_valid.py 