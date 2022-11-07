from typing import Iterable, Iterator, Callable, Dict
import logging
import pdb
import numpy as np 
from overrides import overrides
import pathlib

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, ArrayField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.data.dataset_readers.calflow_parsing.tree_dst_graph import TreeDSTGraph
from miso.data.dataset_readers.calflow import CalFlowDatasetReader
from miso.data.tokenizers import   MisoTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("tree_dst")
class TreeDSTDatasetReader(CalFlowDatasetReader):

    @overrides
    def _read(self, path: str) -> Iterable[Instance]:

        logger.info("Reading calflow data from: %s", path)
        skipped = 0
        source_path = path + ".src_tok"
        target_path = path + ".tgt"

        line_idx_path = path + ".idx"
        if pathlib.Path(line_idx_path).exists():
            with open(source_path) as source_f, open(target_path) as target_f, open(line_idx_path) as line_idx_f:
                for i, (src_line, tgt_line, line_idx) in enumerate(zip(source_f, target_f, line_idx_f)): 
                    graph = TreeDSTGraph(src_str = src_line, 
                                        tgt_str = tgt_line,
                                        use_program = self.use_program,
                                        use_agent_utterance = self.use_agent_utterance,
                                        use_context = self.use_context,
                                        line_idx = line_idx.strip(),
                                        fxn_of_interest = self.fxn_of_interest)
                                        

                    t2i = self.text_to_instance(graph)
                    if t2i is None:
                        skipped += 1
                        continue
                    if self.line_limit is not None:
                        if i > self.line_limit:
                            break
                    yield t2i

        else:
            with open(source_path) as source_f, open(target_path) as target_f:
                for i, (src_line, tgt_line) in enumerate(zip(source_f, target_f)): 
                    graph = TreeDSTGraph(src_str = src_line, 
                                        tgt_str = tgt_line,
                                        use_program = self.use_program,
                                        use_agent_utterance = self.use_agent_utterance,
                                        use_context = self.use_context,
                                        fxn_of_interest = self.fxn_of_interest) 

                    t2i = self.text_to_instance(graph)
                    if t2i is None:
                        skipped += 1
                        continue
                    if self.line_limit is not None:
                        if i > self.line_limit:
                            break
                    yield t2i