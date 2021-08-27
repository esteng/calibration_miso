import pytest
import sys 
import os 

test_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from allennlp.commands.train import Train, train_model_from_file

from utils import assert_successful_overfit, read_metrics, setup_checkpointing_and_args

def test_calflow_lstm_base():
    config_path = os.path.join(test_path, "configs", "overfit_calflow_base.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_calflow_lstm_base.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_exact_match": 100.0}) 

def test_calflow_lstm_context():
    config_path = os.path.join(test_path, "configs", "overfit_calflow_program.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_calflow_lstm_context.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_exact_match": 100.0}) 
