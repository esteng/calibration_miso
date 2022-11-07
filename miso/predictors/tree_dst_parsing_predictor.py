
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
from miso.nn.nucleus_beam_search import CalibratedBeamSearch

logger = logging.getLogger(__name__) 


@Predictor.register("tree_dst_parsing")
class TreeDSTParsingPredictor(CalflowParsingPredictor):


    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        # function hijacked from parent class to return a decomp arborescence instead of printing a line 
        src_str = " ".join(outputs['src_str'])
        pdb.set_trace()
        pred_graph = TreeDSTGraph.from_prediction(src_str,
                                                outputs['nodes'], 
                                                outputs['node_indices'], 
                                                outputs['edge_heads'], 
                                                outputs['edge_types']) 
        return pred_graph.tgt_str


@Predictor.register("tree_dst_parsing_calibrated")
class CalibratedTreeDSTParsingPredictor(TreeDSTParsingPredictor):

    @overrides
    def dump_line(self, outputs: JsonDict, top_k: bool = False) -> str:
        # function hijacked from parent class to return a decomp arborescence instead of printing a line 
        src_str = " ".join(outputs['src_str'])

        # add node probabilities to the nodes in the graph so we can recover them later 
        pred_graph = TreeDSTGraph.from_prediction(src_str,
                                                outputs['nodes'], 
                                                outputs['node_indices'], 
                                                outputs['edge_heads'], 
                                                outputs['edge_types'],
                                                node_probs=outputs['node_probs']) 
        to_ret = {"tgt_str": pred_graph.tgt_str,
                    "expression_probs": pred_graph.expression_probs, 
                    "src_str": src_str,
                    "line_idx": outputs['line_idx']}
        to_ret = json.dumps(to_ret)


        return to_ret

    @overrides 
    def predict_batch_instance(self, 
                                instances: List[Instance], 
                                oracle: bool = False, 
                                top_k_beam_search: bool = False, 
                                top_k_beam_search_hitl = False, 
                                top_k: int = 1,
                                hitl_threshold: float = 0.8) -> List[JsonDict]:
        # set oracle flag for vanilla parsing analysis 
        self._model.oracle = oracle
        self._model.top_k_beam_search = top_k_beam_search
        self._model.top_k_beam_search_hitl = top_k_beam_search_hitl
        self._model.top_k = top_k
        self._model.hitl_threshold = hitl_threshold
        self._model._beam_search = CalibratedBeamSearch(self._model._vocab_eos_index, self._model._max_decoding_steps, top_k)
        self._model._beam_size = top_k
        outputs = self._model.forward_on_instances(instances)
        return [self.dump_line(line, top_k=True) for line in outputs]