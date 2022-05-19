## Intent recognition
In addition to semantic parsing, we explore intent recognition for the [nlu_evaluation_data](https://huggingface.co/datasets/nlu_evaluation_data) dataset. 

The intent recognition data is available via huggingface datasets. 

## Files
The following files are relevant to intent recognition: 

- `data.py` reads, loads, and splits the data 
- `main.py` runs the model, loading commandline arguments. For more info, please run `python main.py --help` 
- `model.py` contains the intent recognition model, which is just a classifier layer on top of BERT 
- `dro_loss.py` implements group DRO loss for intent recognition
- `extract_difficult.py` extracts difficult examples from the predicted and reference data for later analysis. 

