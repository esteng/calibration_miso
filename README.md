# README 
Code for: When More Data Hurts: A Troubling Quirk in Developing Broad-Coverage Natural Language Understanding Systems 

Author: Elias Stengel-Eskin
Personal Email: elias.stengel@gmail.com


## Downloading Data
The first step to replicating experiments is to download the data.

From the project home directory:
```
mkdir -p data 
cd data
wget LINK
unzip LINK
```

## Downloading models 

TODO 

## Training models
Training a model requires the following environment variables: 
1. `CHECKPOINT_DIR`: the directory where the output files will be stored 
2. `TRAINING_CONFIG`: the path to the training jsonnet config. 

For additional details, see ![miso_docs/TRAINING.md](miso_docs/TRAINING.md) 

## Testing models 
The following environment variables need to set:
1. `CHECKPOINT_DIR`: the directory containing a subdirectory `ckpt`, which contains an archive `model.tar.gz`. If training is interrupted or canceled, the archive may be missing. It can be created manually by the following commands: 
```cd $CHECKPOINT_DIR/ckpt
cp best.th weights.th 
tar -czvf model.tar.gz weights.th config.json vocabulary
```
2. `TEST_DATA` is the path to the test data *without the extension*. An example would be `TEST_DATA=data/smcalflow.agent.data/dev_valid`. 
3. `FXN` is the function of interest. Example: `FXN=FindManager` 

The model can then be tested using `./experiments/calflow.sh -a eval_fxn`  

## File Organization 
Important directories: 
- `miso`: contains all the parsing code for the different MISO models 
- `scripts`: contains helper scripts for analysis and creating config files/data splits 
- `experiments`: contains bash files for running MISO parser (see MISO_README.md for more details) 

The main change between different `.jsonnet` files is the data path at the top. This points the model to the correct data split to use, e.g. `data/smcalflow_samples_curated/FindManager/5000_100/` 
points the model to the 5000 train sample subset with 100 FindManager examples. 
The assumption is that each experiment has a jsonnet file.
For example, the experiment which trains a transformer model with the `seed=12` for the 5000-100 FindManager corresponds to the `.jsonnet` file `miso/training_configs/calflow_transformer/FindManager/12_seed/5000_100.jsonnet`. 
In the released configs, the data dir argument is an environment variable 


## Important Scripts
- `scripts/sample_functions.py`: samples functions (e.g. FindManager) to create the different splits. Can be used to manually curate examples. 
- `scripts/make_subsamples.sh`: iteratively runs sampling for each split (5000-max), curating the first one and then using those examples later. 
- `scripts/make_subsamples_uncurated.sh`: same idea, but doesn't require curation (for non-100 splits, no curation is done).
- `scripts/make_configs.py`: can be used to modify a base jsonnet config to change the path to the split
- `scripts/prepare_data.sh`: Data is assumed to be pre-processed according to [Task Oriented Parsing as Dataflow Synethesis](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis) instructions. This is a modified version of the instructions in the README there to include agent utterances and previous user turns. 
- `scripts/collect_results.py`: script to collect exact match results from predictions, written to `CHECKPOINT_DIR/translate_output`. Aggregates all scores into a csv specified as an arg.
- `experiments/calflow.sh`: main training/testing commands for calflow

## Other scripts 
- `scripts/split_valid.py`: splits all valid dialogs into dev and test subsets. 
- `scripts/error_analysis.py`: for a given function, analyze predicted plans into 3 groups: correct predictions, incorrect examples wihtout the function, incorrect examples with the function. 
- `scripts/oversample.py`: either over-sample examples for a given function (e.g. turn 5000-100 FindManager into 5000-200 by doubling the 100 FindManager examples) or over-sample the rest of the training data to get a split of e.g. 200k-100 
where 200k is upsampled from the max setting. 

## Training models 
Models can be trained locally using `experiments/calflow.sh`. 
`experiments/calflow.sh` expects the following environment variables to be set: `CHECKPOINT_DIR`, `TRAINING_CONFIG`, and `DATA_ROOT`. `DATA_ROOT` is the location where you downloaded the data. 
The former points to a directory where the model will store checkpoints. The latter is a `.jsonnet` config that will be read by AllenNLP. 
Optionally, the `FXN` variable can also be set, for function-specific evaluation. 

Model checkpoints and logs will be written to `CHECKPOINT_DIR/ckpt`. Decoded outputs will be written to `CHECKPOINT_DIR/translate_output/<split}>.tgt` 

