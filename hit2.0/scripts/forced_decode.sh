#!/bin/bash

export TEST_DATA=$1
export FXN=None
export CHECKPOINT_DIR=/brtx/604-nvme1/estengel/calflow_calibration/miso/tune_roberta_tok_fix_benchclamp_data/

./experiments/calflow.sh -a log_short
