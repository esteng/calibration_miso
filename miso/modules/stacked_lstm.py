# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple
import torch
from torch.nn.utils.rnn import PackedSequence

from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.common.registrable import Registrable
from allennlp.common.checks import ConfigurationError


class StackedLstm(torch.nn.Module, Registrable):
    """
    A standard stacked LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular LSTM is the application of
    variational dropout to the hidden states of the LSTM.
    Note that this will be slower, as it doesn't use CUDNN.
    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 recurrent_dropout_probability: float = 0.0,
                 use_highway: bool = True) -> None:
        super(StackedLstm, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):

            layer = AugmentedLstm(lstm_input_size, hidden_size,
                                  go_forward=True,
                                  recurrent_dropout_probability=recurrent_dropout_probability,
                                  use_highway=use_highway,
                                  use_input_projection_bias=False)
            lstm_input_size = hidden_size
            self.add_module('layer_{}'.format(layer_index), layer)
            layers.append(layer)

        self.lstm_layers = layers

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).
        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))

        output_sequence = inputs
        final_states = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'layer_{}'.format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            output_sequence, final_state = forward_layer(output_sequence, state)
            final_states.append(final_state)

        final_state_tuple = [torch.cat(state_list, 0) for state_list in zip(*final_states)]
        return output_sequence, final_state_tuple
