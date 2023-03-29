from typing import Dict, Optional

import overrides 
import torch 
from transformers import AutoModel 

from miso.modules.decoders.decoder import MisoDecoder

# TODO (elias): Wrap MisoTransformerDecoder to load weights from T5 model 

# TODO (elias): should this really inherit from MisoTransformerDecoder?
class T5Decoder(MisoDecoder):
    def __init__(self, model_name: str):
        super().__init__()
        # super().__init__(input_size=None,
        #                 hidden_size=None,
        #                 decoder_layer=None,
        #                 num_layers=None,
        #                 source_attention_layer=None,
        #                 target_attention_layer=None,
        #                 norm=None,
        #                 dropout=None,
        #                 use_coverage=None)
        self._model = AutoModel.from_pretrained(model_name)
        # extract only the decoder part of the model 
        self._decoder = self._model.decoder

    @overrides 
    def forward(self,
                inputs: torch.Tensor,
                source_memory_bank: torch.Tensor,
                source_mask: torch.Tensor,
                target_mask: torch.Tensor,
                is_train: bool = True) -> Dict: 
        # we can mostly use T5 code here 
        raise NotImplementedError

    @overrides     
    def one_step_forward(self,
                         inputs: torch.Tensor,
                         source_memory_bank: torch.Tensor,
                         source_mask: torch.Tensor,
                         decoding_step: int = 0,
                         total_decoding_steps: int = 0,
                         coverage: Optional[torch.Tensor] = None) -> Dict:
        raise NotImplementedError 