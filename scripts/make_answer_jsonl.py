import json 
import argparse

# format 
# {"datum_id": {"dialogue_id": "5da3326f-3f56-414e-86e9-f155fb12028f", "turn_index": 0}, "lispress": "(FenceScope)", "program_execution_oracle": {"has_exception": false, "refer_are_correct": true}, "user_utterance": "Change the reservation for tonight to 6 people from 4."}

# /brtx/601-nvme1/estengel/resources/data/tree_dst.agent.data/valid.datum_id has {"dialogue_id": "5da3326f-3f56-414e-86e9-f155fb12028f", "turn_index": 0} 
# tree_dst.agent.data/valid.dataflow_dialogues.jsonl,  "turns" has lispress, program_execution_oracle, user_utterance



def main(args):
    with open(args.datum_id_file) as f1:
        datum_ids = [json.loads(line.strip()) for line in f1]

    with open(args.dataflow_dialogues_file) as f2:
        dataflow_dialogues = [json.loads(line.strip()) for line in f2]

    to_write = []
    dialogues_by_id = {d['dialogue_id']: d for d in dataflow_dialogues}
    for datum_id in datum_ids:
        dialogue = dialogues_by_id[datum_id['dialogue_id']]
        turn = dialogue['turns'][datum_id['turn_index']]
        lispress = turn['lispress']
        program_execution_oracle = turn['program_execution_oracle']
        user_utterance = turn['user_utterance']
        datapoint = {"datum_id": datum_id, "lispress": lispress, "program_execution_oracle": program_execution_oracle, "user_utterance": user_utterance}
        to_write.append(datapoint)

    with open(args.output_file, 'w') as f3:
        for datapoint in to_write:
            f3.write(json.dumps(datapoint) + "\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datum_id_file", type=str, required=True)
    parser.add_argument("--dataflow_dialogues_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)