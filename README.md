# README 
Code for: Calibrated Interpretation: Confidence Estimation in Semantic Parsing

Author: Elias Stengel-Eskin

Personal Email: elias.stengel@gmail.com

## About the repo 
This repo is a fork of [this repo](https://github.com/microsoft/nlu-incremental-symbol-learning), which is itself a fork of a fork of [MISO](https://github.com/esteng/miso_uds) which is a semantic parsing codebase that was released with [Joint Universal Syntactic and Semantic Parsing](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00396/106796/Joint-Universal-Syntactic-and-Semantic-Parsing). 

MISO was built over the course of the following publications:  
- [AMR Parsing as Sequence-to-Graph Transduction, Zhang et al., ACL 2019](https://www.aclweb.org/anthology/P19-1009/)
- [Broad-Coverage Semantic Parsing as Transduction, Zhang et al., EMNLP 2019](https://www.aclweb.org/anthology/D19-1392/)
- [Universal Decompositional Semantic Parsing, Stengel-Eskin et al. ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.746/)
- [Joint Universal Syntactic and Semantic Parsing, Stengel-Eskin et al., TACL 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00396/106796/Joint-Universal-Syntactic-and-Semantic-Parsing)
- [When More Data Hurts: A Troubling Quirk in Developing Broad-Coverage Natural Language Understanding Systems, Stengel-Eskin et al., EMNLP 2022](https://arxiv.org/abs/2205.12228)

It is a flexible sequence-to-graph parsing framework built on top of [allennlp](https://github.com/allenai/allennlp).  


## Easy and Hard splits
The directory `data_subsets` contains the easy and hard splits of TreeDST and SMCalFlow described in the paper. 
These can also be downloaded directly [here](https://nlp.jhu.edu/semantic_parsing_calibration/data_subsets.tar.gz).

# MISO Documentation
## Installation 

All dependencies can be installed with `./install_requirements.sh` 

## Downloading Data
The first step to replicating experiments is to download the data and glove embeddings.

From the project home directory:

```
mkdir -p data 
cd data
# This may take some time 
wget https://veliass.blob.core.windows.net/ifl-data/data_clean.tar.gz
tar -xzvf data_clean.tar.gz 
mv data_clean/* .
rm -r data_clean 
```


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
- `scripts/prepare_data.sh`: Data is assumed to be pre-processed according to [Task Oriented Parsing as Dataflow Synethesis](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis) instructions. This is a modified version of the instructions in the README there to include agent utterances and previous user turns. 
- `experiments/calflow.sh`: main training/testing commands for calflow
- `experiments/tree_dst.sh`: main training/testing commands for TreeDST


## Training Models 
Models can be trained locally using `experiments/calflow.sh`. 
`experiments/calflow.sh` expects the following environment variables to be set: `CHECKPOINT_DIR`, `TRAINING_CONFIG`, and `DATA_ROOT`. `DATA_ROOT` is the location where you downloaded the data. 
The former points to a directory where the model will store checkpoints. The latter is a `.jsonnet` config that will be read by AllenNLP. 
Optionally, the `FXN` variable can also be set, for function-specific evaluation. 

Model checkpoints and logs will be written to `CHECKPOINT_DIR/ckpt`. Decoded outputs will be written to `CHECKPOINT_DIR/translate_output/<split}>.tgt` 


For additional details, see [miso_docs/TRAINING.md](miso_docs/TRAINING.md) 

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

The output at the end will have the following rows: 

```
Exact Match: The overall exact match accuracy of produced and reference programs. 
FXN Coarse: The percentage of programs for which, if FXN is in the reference, it is also in the predicted program. It doesn't matter if the programs match or not. 
FindManager Fine: The percentage of programs with FXN in the reference where the predicted program is an exact match. 
FindManager Precision: The percentage of predicted programs that have FXN in them and also have FXN in the reference program. 
FindManager Recall: Same as Coarse 
FindManager F1: Harmonic mean of precision and recall 
```

## Getting logits
To get the predicted token logits under a forced decode, see the `log_losses` function in `experiments/calflow.sh`. 
To get token-level predicted probabilities without a forced decode, use `eval_calibrate`. 
