
from miso.nn.beam_search import BeamSearch
from dataflow.core.lispress import render_compact
from overrides import overrides
from typing import List, Iterator, Any
import numpy
from contextlib import contextmanager
import json
import logging 
import sys
import pdb 

import numpy as np 
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
from allennlp.common.util import JsonDict

from miso.data.dataset_readers.calflow_parsing.tree_dst_graph import TreeDSTGraph
from miso.predictors.decomp_parsing_predictor import sanitize
from miso.predictors.calflow_parsing_predictor import CalflowParsingPredictor

logger = logging.getLogger(__name__) 


@Predictor.register("tree_dst_parsing")
class TreeDSTParsingPredictor(CalflowParsingPredictor):


    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        # function hijacked from parent class to return a decomp arborescence instead of printing a line 
        src_str = " ".join(outputs['src_str'])
        pred_graph = TreeDSTGraph.from_prediction(src_str,
                                                outputs['nodes'], 
                                                outputs['node_indices'], 
                                                outputs['edge_heads'], 
                                                outputs['edge_types']) 
        return pred_graph.tgt_str

