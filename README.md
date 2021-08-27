# README 
This is a readme for the Incremental Function Learning project 

Author: Elias Stengel-Eskin
Personal Email: elias.stengel@gmail.com

## File Organization 
Important directories: 
- `miso`: contains all the parsing code for the different MISO models 
- `amlt_configs`: contains amulet configs for running jobs 
- `scripts`: contains helper scripts for analysis and creating config files/data splits 
- `analysis`: contains analysis ipynb and scripts 
- `experiments`: contains bash files for running MISO parser (see MISO_README.md for more details) 

Everything is run through amulet, and the scripts assume that your data is organized in the following way. 
All data should be in container mounted as `/mnt/default/resources/data`. I used blobfuse to mount this container locally so that data can easily be added and modified though the CLI.
After modifying data, before running a model, it needs to be uploaded using `amlt upload --config-file <path_to_config>`. Because the data dir is shared between all configs, this only needs
to happen once after modification, rather than for each config. 

The main change between different `.jsonnet` files is the data path at the top. This points the model to the correct data split to use, e.g. `/mnt/default/resources/data/smcalflow_samples_curated/FindManager/5000_100/` 
points to the model to the 5000 train sample subset with 100 FindManager examples. 
The assumption is that each experiment has a jsonnet file, and a corresponding amulet config.
For example, the experiment which trains a transformer model with the `seed=12` for the 5000-100 FindManager split has the amulet config `amlt_configs/transformer/FindManager_12_seed/5000_100.yaml` which points 
to the `.jsonnet` file `miso/training_configs/calflow_transformer/FindManager/12_seed/5000_100.jsonnet`. 
The amulet config also sets the `CHECKPOINT_DIR` var, which is where the model will be stored. This container should also be mounted with blobfuse. It should be mounted to `/home/<user>/amlt_models/`. 


## Important Scripts
- `scripts/sample_functions.py`: samples functions (e.g. FindManager) to create the different splits. Can be used to manually curate examples. 
- `scripts/make_subsamples.sh`: iteratively runs sampling for each split (5000-max), curating the first one and then using those examples later. 
- `scripts/make_subsamples_uncurated.sh`: same idea, but doesn't require curation (for non-100 splits, no curation is done).
- `scripts/make_configs.py`: can be used to modify a base jsonnet config to change the path to the split
- `scripts/prepare_data.sh`: Data is assumed to be pre-processed according to [Task Oriented Parsing as Dataflow Synethesis](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis) instructions. This is a modified version of the instructions in the README there to include agent utterances and previous user turns. 
- `scripts/collect_results.py`: script to collect exact match results from predictions, written to `CHECKPOINT_DIR/translate_output`. Aggregates all scores into a csv specified as an arg.
- `scripts/make_line_nums.py`: makes a .idx file given a dir with a train and test file, which is required for batching minimal pairs. A .idx file just has the line idxs (0-N) per line, but having it stored separately makes batching easier. 
- `scripts/make_lookup_table.py`: used to combine minimal pairs produced by a model from manually-annotated instances with their corresponding input. To be used with `miso.data.iterators.minimal_pair.RealMinimalPairIterator`. 
- `submit_amlt_dangerous.sh`: submit all jobs across a split for a given function, seed, and model. Called `dangerous` because it will over-write previously submitted jobs of the same name automatically to avoid lots of prompting per submission. 
- `decode_amlt_dangerous.sh`: decode test predictions for a given function, seed, and model. 
- `experiments/calflow.sh`: main training/testing commands for calflow

## Other scripts 
- `scripts/annotate_minimal_pairs.py`: helper annotation script to modify minimal pairs manually.
- `scripts/split_valid.py`: splits all valid dialogs into dev and test subsets. 
- `scripts/leven.py`: levenshtein utils to do analysis between anonymized plans in correct and incorrect set and min distance to a train example
- `scripts/error_analysis.py`: for a given function, analyze predicted plans into 3 groups: correct predictions, incorrect examples wihtout the function, incorrect examples with the function. 
- `scripts/oversample.py`: either over-sample examples for a given function (e.g. turn 5000-100 FindManager into 5000-200 by doubling the 100 FindManager examples) or over-sample the rest of the training data to get a split of e.g. 200k-100 
where 200k is upsampled from the max setting. 
- `decode_amlt_beam.sh`: decode top 100 test predictions for given function, seed, model for beam analysis 
- `decode_amlt_prob.sh`: forced-decode all test examples and save whole output distribution at each timestep for a function, seed, and model type

## Training models 
Models can be trained locally using `experiments/calflow.sh` or on Azure using amulet. The second option is more common.
`experiments/calflow.sh` expects the following environment variables to be set: `CHECKPOINT_DIR` and `TRAINING_CONFIG`. 
The former points to a directory where the model will store checkpoints. The latter is a `.jsonnet` config that will be read by AllenNLP. 
Optionally, the `FXN` variable can also be set, for function-specific evaluation. 

Each amulet config has several jobs. The `train` job should run training and then decoding for dev and test. In case the decoding jobs fail or need 
to be re-run, there is a `decode` and `decode_test` job that will run the decoding separately. 
Model checkpoints and logs will be written to `CHECKPOINT_DIR/ckpt`. Decoded outputs will be written to `CHECKPOINT_DIR/translate_output/<split}>.tgt` 

## Minimal pair utils
- `minimal_pair_utils` contains a set of tools for extracting minimal pairs. 
- `construct.py` is the main file, which builds a generated lookup table to be used with `miso.data.iterators.minimal_pair.GeneratedRealMinimalPairIterator`. 
- `evaluate.py` can be used to manually inspect the quality of the output pairs. 
- `extract_fxn_lines.py`: gets lines from a test set which contain the desired function (for speed)
- `extract_all_function_lines.sh`: repeats extraction across all splits 
- `get_names.py`: gets names from train data for use in `construct.py`. Not used in practice.
- `levenshtein.py`: does levenshtein computation between test and train utterances. 
- `mutations.py`: Set of mutations that can be applied to input. In practice, only identity is used. 

