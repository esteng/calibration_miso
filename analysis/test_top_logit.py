import json
import re 
import pdb 
import numpy as np 
import sys 
import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter, defaultdict
from scipy import stats

from calibration_metric.vis.calibration_plot import plot_df, get_df_from_file
from calibration_metric.metric import ECEMetric
from calibration_metric.utils.reader import TopKTopLogitFormatSequenceReader

plt.rcParams["font.family"] = "Nimbus Roman"

from calibration_utils import (read_benchclamp_file, 
                                get_probs_and_accs_sql) 

sys.path.insert(0, "/home/estengel/semantic_parsing_with_constrained_lm/src/")

path = "/brtx/604-nvme2/estengel/calflow_calibration/benchclamp/1.0/t5-small-lm-adapt_spider_past_none_db_val_all_0.0001/checkpoint-10000//outputs/test_all.logits"
df, ece = get_df_from_file(path, n_bins=10, binning_strategy="adaptive", ignore_tokens=['"', "'"], reader_cls = TopKTopLogitFormatSequenceReader, reader_kwargs = {"k": 3})
