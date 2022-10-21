# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from miso.nn.beam_search import BeamSearch
# from miso.nn.calibrated_beam_search import CalibratedBeamSearch
from miso.nn.nucleus_beam_search import CalibratedBeamSearch
from dataflow.core.lispress import render_compact
from overrides import overrides
from typing import List, Iterator, Any
import numpy
from contextlib import contextmanager
import json
import logging 
import sys
import pdb 
from networkx.readwrite import json_graph

import numpy as np 
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
from allennlp.common.util import JsonDict

from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from miso.data.dataset_readers.calflow_parsing.calflow_sequence import CalFlowSequence
from miso.predictors.decomp_parsing_predictor import sanitize

logger = logging.getLogger(__name__) 


@Predictor.register("calflow_parsing")
class CalflowParsingPredictor(Predictor):

    @overrides
    def dump_line(self, outputs: JsonDict, top_k: bool = False) -> str:
        # function hijacked from parent class to return a decomp arborescence instead of printing a line 
        src_str = " ".join(outputs['src_str'])

        if "edge_heads" in outputs.keys():
            pred_graph = CalFlowGraph.from_prediction(src_str,
                                                    outputs['nodes'], 
                                                    outputs['node_indices'], 
                                                    outputs['edge_heads'], 
                                                    outputs['edge_types']) 
            to_ret = pred_graph.tgt_str

        elif len(outputs.keys()) == 2 and sorted(list(outputs.keys()))[0] == "output": 
            # basic classification 
            pred_idx = np.argmax(outputs['output'], axis=0)
            pred_str = f"Func{pred_idx+1}"
            to_ret = pred_str 
        else:

            pred_str = CalFlowSequence.from_prediction(outputs['nodes'])
            to_ret = pred_str
        if top_k:
            return "<SEP>".join([src_str, to_ret])
        return to_ret


    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, 
                                instances: List[Instance], 
                                oracle: bool = False, 
                                top_k_beam_search: bool = False, 
                                top_k_beam_search_hitl: bool = False, 
                                top_k: int = 1) -> List[JsonDict]:
        # set oracle flag for vanilla parsing analysis 
        self._model.oracle = oracle
        self._model.top_k_beam_search = top_k_beam_search
        self._model.top_k_beam_search_hitl = top_k_beam_search_hitl
        self._model.top_k = top_k
        if top_k > 1:
            self._model._beam_search = BeamSearch(self._model._vocab_eos_index, self._model._max_decoding_steps, top_k)
            self._model._beam_size = top_k
        outputs = self._model.forward_on_instances(instances)
        if oracle: 
            return self.organize_forced_decode(instances, outputs) 
        if top_k_beam_search:
            return [self.dump_line(line) for line in outputs]
        if top_k_beam_search_hitl:
            return [self.dump_line(line) for line in outputs]
        return sanitize(outputs)

    def organize_forced_decode(self, instances, outputs):
        to_ret = []
        for instance, output in zip(instances, outputs):
            try:
                prob_dist = output['prob_dist'].tolist()
            except AttributeError:
                prob_dist = output['prob_dist']
            source_tokens = instance['source_tokens'].tokens
            left_context = instance['target_tokens'].tokens
            next_token = instance['generation_outputs'].tokens[1:]
            to_ret.append({"source_tokens": source_tokens, "left_context": left_context, "next_token": next_token, "prob_dist": prob_dist})
        assert(len(to_ret) == 1)
        return to_ret[0]

@Predictor.register("calflow_parsing_calibrated")
class CalibratedCalflowParsingPredictor(CalflowParsingPredictor):

    @overrides
    def dump_line(self, outputs: JsonDict, top_k: bool = False) -> str:
        # function hijacked from parent class to return a decomp arborescence instead of printing a line 
        src_str = " ".join(outputs['src_str'])

        # add node probabilities to the nodes in the graph so we can recover them later 
        pred_graph = CalFlowGraph.from_prediction(src_str,
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
                                top_k: int = 1) -> List[JsonDict]:
        # set oracle flag for vanilla parsing analysis 
        self._model.oracle = oracle
        self._model.top_k_beam_search = top_k_beam_search
        self._model.top_k_beam_search_hitl = top_k_beam_search_hitl
        self._model.top_k = top_k
        self._model._beam_search = CalibratedBeamSearch(self._model._vocab_eos_index, self._model._max_decoding_steps, top_k)
        self._model._beam_size = top_k
        outputs = self._model.forward_on_instances(instances)
        return [self.dump_line(line, top_k=True) for line in outputs]