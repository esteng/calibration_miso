import json
import re 
import pdb 
import numpy as np 
import sys 
import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter, defaultdict
from scipy import stats

from calibration_metric.vis.calibration_plot import plot_df
from calibration_metric.metric import ECEMetric

plt.rcParams["font.family"] = "Nimbus Roman"

from calibration_utils import (read_benchclamp_file, 
                                get_probs_and_accs_sql) 

sys.path.insert(0, "/home/estengel/semantic_parsing_with_constrained_lm/src/")

bart_data = read_benchclamp_file("/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/bart-large_spider_past_none_db_val_all_0.0001_5000_test_eval_unconstrained-beam_bs_5/model_outputs.20230208T031316.jsonl")
# bart_min_probs, bart_mean_probs, bart_accs = get_probs_and_accs_benchclamp(bart_data) 
# bart_min_probs, bart_mean_probs, bart_accs = get_probs_and_accs_sql(bart_data[0:100], "/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/Spider/test_all.jsonl") 

# pdb.set_trace()
mbart_min_probs, mbart_mean_probs, mbart_accs = get_probs_and_accs_sql(bart_data[0:100], "/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/Spider/test_all.jsonl", n_workers = 10) 
pdb.set_trace()

# t5_data = read_benchclamp_file("/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/1.0/t5-base-lm-adapt_spider_past_none_db_val_all_0.0001_10000_test_eval_unconstrained-beam_bs_5/model_outputs.20230206T093954.jsonl") 
# t5_min_probs, t5_mean_probs, t5_accs = get_probs_and_accs_benchclamp(t5_data)
# t5_min_probs, t5_mean_probs, t5_accs = get_probs_and_accs_sql(t5_data)
