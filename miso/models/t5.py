# from miso.models.calflow_parser import CalFlowParser
# from typing import List, Dict, Tuple, Any 
# import logging
# from collections import OrderedDict

# import pdb 
# from overrides import overrides
# import torch
# import numpy as np

# from allennlp.data import Token, Instance, Vocabulary
# from allennlp.data.fields import TextField
# from allennlp.data.dataset import Batch
# from allennlp.models import Model
# from allennlp.modules import TextFieldEmbedder,  Seq2SeqEncoder
# from allennlp.nn import util
# from allennlp.nn.util import  get_text_field_mask
# from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
# from allennlp.common.util import START_SYMBOL, END_SYMBOL
# from allennlp.training.metrics import SequenceAccuracy

# from transformers import AutoModel, T5ForConditionalGeneration, T5Tokenizer, T5Config

# from miso.modules.label_smoothing import LabelSmoothing

# @Model.register("t5_seq2seq_parser")
# class T5Parser(Model):
#     def __init__(self, 
#                  vocab: Vocabulary,
#                  model_name: str,
#                  # misc
#                  label_smoothing: LabelSmoothing,
#                  target_output_namespace: str,
#                  dropout: float = 0.0,
#                  beam_size: int = 5,
#                  max_decoding_steps: int = 50,
#                  training: bool = True,
#                  eps: float = 1e-20):
#         super().__init__(vocab=vocab) 

#         self.model = T5ForConditionalGeneration.from_pretrained(model_name)
#         self.label_smoothing = label_smoothing
#         self.target_output_namespace = target_output_namespace
#         self.dropout = torch.nn.Dropout(dropout)
#         self.beam_size = beam_size
#         self.max_decoding_steps = max_decoding_steps
#         self.eps = eps
#         self.training = training 

#     @overrides
#     def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#         raise NotImplementedError

#     @overrides
#     def forward(self, **raw_inputs: Dict) -> Dict:
#         inputs = self._prepare_inputs(raw_inputs)
#         if self.training:
#             return self._training_forward(inputs)
#         else:
#             return self._test_forward(inputs)

#     @overrides
#     def _prepare_inputs(self, raw_inputs):
#         inputs = raw_inputs.copy()
#         return inputs

#     def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
#         self.model(**inputs)
#         raise NotImplementedError


#     def _test_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
#         raise NotImplementedError