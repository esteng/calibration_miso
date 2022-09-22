from typing import List, Callable, Tuple, Dict, Any
import warnings
from overrides import overrides

import pdb 
import torch

from allennlp.common.checks import ConfigurationError
import logging

from miso.nn.beam_search import BeamSearch

logger = logging.getLogger(__name__) 

StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
AuxiliaryType = Dict[str, List[Any]]  # pylint: disable=invalid-name
StepFunctionType = Callable[[torch.Tensor, StateType, AuxiliaryType], Tuple[torch.Tensor, StateType, AuxiliaryType]]  # pylint: disable=invalid-name


class CalibratedBeamSearch(BeamSearch):
    """
    Implements the beam search algorithm for decoding the most likely sequences.

    Parameters
    ----------
    end_index : ``int``
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : ``int``, optional (default = 50)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : ``int``, optional (default = 10)
        The width of the beam used.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to ``beam_size``. Setting this parameter
        to a number smaller than ``beam_size`` may give better results, as it can introduce
        more diversity into the search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10,
                 per_node_beam_size: int = None,
                 confidence_threshold: float = 0.70) -> None:
        super().__init__(end_index, max_steps, beam_size, per_node_beam_size)
        self.confidence_threshold = confidence_threshold

    @overrides
    def search(self,
               start_predictions: torch.Tensor,
               start_state: StateType,
               auxiliaries: AuxiliaryType,
               step: StepFunctionType,
               tracked_state_name: str,
               tracked_auxiliary_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[Any]]]:
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.

        Notes
        -----
        If your step function returns ``-inf`` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have ``-inf`` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from ``search``
        and potentially discard sequences with non-finite log probability.

        Parameters
        ----------
        start_predictions : ``torch.Tensor``
            A tensor containing the initial predictions with shape ``(batch_size,)``.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.
        start_state : ``StateType``
            The initial state passed to the ``step`` function. Each value of the state dict
            should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
            number of dimensions.
        auxiliaries: ``AuxiliaryType``
            The auxiliaries passed to the ``step`` function. Each value of the auxiliary dict
            should be a list of batch_size items.
        step : ``StepFunctionType``
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.
        tracked_state_name: ``str``
            The tracked state name.
        tracked_auxiliary_name: ``str``
            The tracked auxiliary name.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[Any]]
            Tuple of ``(predictions, tracked_states, log_probabilities)``, where
            ``predictions`` has shape ``(batch_size, beam_size, max_steps)``,
            ``tracked_states`` has shape ``(batch_size, beam_size, max_steps, *)``,
            ``log_probabilities`` has shape ``(batch_size, beam_size)``, and
            ``tracked_auxiliaries`` has shape ``(beam_size, batch_size)``.
        """
        batch_size = start_predictions.size()[0]

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []
        # List of (batch_size, beam_size, *) tensors.
        tracked_states: List[torch.Tensor] = []
        # A 2d array: (beam_size, batch_size)
        tracked_auxiliaries: List[List[Any]] = [[None for _ in range(batch_size)] for _ in range(self.beam_size)]

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state, auxiliaries = step(start_predictions, start_state, auxiliaries, 1)

        num_classes = start_class_log_probabilities.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise ConfigurationError(f"Target vocab size ({num_classes:d}) too small "
                                     f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                                     f"Please decrease beam_size or per_node_beam_size.")

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probabilities, start_predicted_classes = \
                start_class_log_probabilities.topk(self.beam_size)

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
                (batch_size * self.beam_size, num_classes),
                float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor.\
                    unsqueeze(1).\
                    expand(batch_size, self.beam_size, *last_dims).\
                    reshape(batch_size * self.beam_size, *last_dims)

        # Set the same auxiliaries for each element in the beam.
        for key, aux in auxiliaries.items():
            new_aux = []
            for element in aux:
                new_aux += [element.copy() for _ in range(self.beam_size)]
            auxiliaries[key] = new_aux

        tracked_state = state[tracked_state_name]
        _, *last_dims = tracked_state.size()
        # shape: [(batch_size, beam_size, *)]
        tracked_states.append(tracked_state.reshape(batch_size, self.beam_size, *last_dims))
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn("Empty sequences predicted. You may want to increase the beam size or ensure "
                          "your step function is working properly.",
                          RuntimeWarning)
            # shape: (batch_size * beam_size)
            tracked_auxiliary = auxiliaries[tracked_auxiliary_name]
            # shape: (batch_size, beam_size)
            for beam_index in range(self.beam_size):
                for i in range(beam_index, len(tracked_auxiliary), self.beam_size):
                    tracked_auxiliaries[beam_index][i // self.beam_size] = tracked_auxiliary[i]
            return (start_predicted_classes.unsqueeze(-1),
                    tracked_states[-1].unsqueeze(2),
                    start_top_log_probabilities,
                    tracked_auxiliaries)

        size_multiplier = 1
        copy_state = state.copy() 

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size  * size_multiplier)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)

            # multiply state 
            state = copy_state 
            class_log_probabilities, state, auxiliaries = step(last_predictions, state,  auxiliaries, size_multiplier)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                    batch_size * self.beam_size * size_multiplier,
                    num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size * size_multiplier, num_classes)
            log_probs_after_end = log_probs_after_end[0:self.beam_size].repeat((size_multiplier, 1))
            cleaned_log_probabilities = torch.where(
                    last_predictions_expanded == self._end_index,
                    log_probs_after_end,
                    class_log_probabilities
            )

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probabilities, predicted_classes = \
                cleaned_log_probabilities.topk(self.per_node_beam_size)

            # NOTE (elias): add a check for the confidence scores 
            # shape: (batch_size * beam_size)
            is_low_confidence = torch.exp(top_log_probabilities.max(dim=-1)[0]) < self.confidence_threshold

            # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            try:
                expanded_last_log_probabilities = last_log_probabilities.\
                        unsqueeze(2).\
                        expand(batch_size, self.beam_size * size_multiplier, self.per_node_beam_size).\
                        reshape(batch_size * self.beam_size * size_multiplier, self.per_node_beam_size)
            except RuntimeError:
                pdb.set_trace()
            # Switch beam around so that the top hypothesis from each low confidence option is kept 
            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            # NOTE (elias): once everything is tested we can get rid of this first check 
            # if all predictions are high confidence, do nothing different 
            if not torch.any(is_low_confidence):
                # shape: (batch_size, beam_size * per_node_beam_size)
                reshaped_summed = summed_top_log_probabilities.\
                        reshape(batch_size, self.beam_size * size_multiplier * self.per_node_beam_size)

                # shape: (batch_size, beam_size * per_node_beam_size)
                reshaped_predicted_classes = predicted_classes.\
                        reshape(batch_size, self.beam_size * size_multiplier * self.per_node_beam_size)

                # Keep only the top `beam_size` beam indices.
                # shape: (batch_size, beam_size), (batch_size, beam_size)
                restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)

                # Use the beam indices to extract the corresponding classes.
                # shape: (batch_size, beam_size)
                restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)

                copy_state = state 

                # need to repeat 
                restricted_beam_log_probs = restricted_beam_log_probs.repeat((1, size_multiplier))
                restricted_beam_indices = restricted_beam_indices.repeat((1, size_multiplier))
                restricted_predicted_classes = restricted_predicted_classes.repeat((1, size_multiplier))
                n_nonconfident = 0
                n_confident = self.beam_size
            # if top prediction is low confidence, we need to keep all of the possibilities for low confidence token, expand that beam 
            else:
                # two paths: confident get same treatement as before 
                n_confident = torch.sum(~is_low_confidence)
                restricted_predicted_classes = None
                # only do if there are any confident, otherwise we get an error
                if n_confident > 0:
                    confident_summed_top_log_probs = summed_top_log_probabilities[~is_low_confidence]
                    confident_predicted_classes = predicted_classes[~is_low_confidence]
                    # shape: (batch_size, n_confident * per_node_beam_size)
                    reshaped_confident_summed = confident_summed_top_log_probs.\
                            reshape(batch_size, n_confident * self.per_node_beam_size)

                    # shape: (batch_size, n_confident * per_node_beam_size)
                    reshaped_confident_predicted_classes = confident_predicted_classes.\
                            reshape(batch_size, n_confident * self.per_node_beam_size)

                    # Keep only the top `beam_size` beam indices.
                    # if there are any confident cases, these should be included by default, so use the general variable here  
                    # shape: (batch_size, beam_size), (batch_size, beam_size)
                    restricted_beam_log_probs, restricted_beam_indices = reshaped_confident_summed.topk(self.beam_size)

                    # Use the beam indices to extract the corresponding classes.
                    # shape: (batch_size, beam_size)
                    restricted_predicted_classes = reshaped_confident_predicted_classes.gather(1, restricted_beam_indices)

                # low confidence tokens get expanded in a separate buffer 
                n_nonconfident = torch.sum(is_low_confidence)
                if n_nonconfident > 0:
                    nonconfident_summed_top_log_probs = summed_top_log_probabilities[is_low_confidence]
                    nonconfident_predicted_classes = predicted_classes[is_low_confidence]
                    # shape: (batch_size, n_nonconfident * per_node_beam_size)
                    reshaped_nonconfident_summed = nonconfident_summed_top_log_probs.\
                            reshape(batch_size, n_nonconfident * self.per_node_beam_size)

                    # shape: (batch_size, n_nonconfident * per_node_beam_size)
                    reshaped_nonconfident_predicted_classes = nonconfident_predicted_classes.\
                            reshape(batch_size, n_nonconfident * self.per_node_beam_size)

                    # Keep all per_node_beam_size indices here so we can later expand
                    nonconfident_beam_log_probs, nonconfident_beam_indices = reshaped_nonconfident_summed.topk(self.per_node_beam_size * n_nonconfident)

                    # Keep only the top `beam_size` beam indices for the main buffer 
                    # shape: (batch_size, beam_size), (batch_size, beam_size)
                    # restricted_nonconfident_beam_log_probs, restricted_nonconfident_beam_indices = reshaped_nonconfident_summed.topk(self.beam_size)

                    # Use the beam indices to extract the corresponding classes.
                    # shape: (batch_size, beam_size)
                    # restricted_nonconfident_predicted_classes = reshaped_confident_predicted_classes.gather(1, restricted_nonconfident_beam_indices)
                    predicted_classes = reshaped_nonconfident_predicted_classes
                    if restricted_predicted_classes is None:
                        restricted_predicted_classes = predicted_classes
                        restricted_beam_log_probs = nonconfident_beam_log_probs
                        restricted_beam_indices = nonconfident_beam_indices
                    else:
                        restricted_predicted_classes = torch.cat([restricted_predicted_classes, predicted_classes], dim=1)
                        restricted_beam_log_probs = torch.cat([restricted_beam_log_probs, nonconfident_beam_log_probs], dim=1)
                        restricted_beam_indices = torch.cat([restricted_beam_indices, nonconfident_beam_indices], dim=1)

                    size_multiplier = restricted_predicted_classes.shape[1] // self.beam_size
                    copy_state = {}
                    for key, state_tensor in state.items():
                        # get the original state tensor before repeating 
                        state_tensor = state_tensor[0:self.beam_size * batch_size]

                        if len(state_tensor.shape) == 3:
                            new_state_tensor = state_tensor.repeat((size_multiplier, 1, 1))
                        elif len(state_tensor.shape) == 2:
                            new_state_tensor = state_tensor.repeat((size_multiplier, 1)) 
                        # need to double because we're expanding the beam
                        # TODO (elias): do we need to clone? 
                        copy_state[key] = new_state_tensor.clone()

            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices // (self.per_node_beam_size * size_multiplier)
            # print(f"backpointer: {backpointer.shape}")
            backpointer = backpointer.long() 
            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in copy_state.items():
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                try:
                    expanded_backpointer = backpointer.\
                            view(batch_size, self.beam_size * size_multiplier, *([1] * len(last_dims))).\
                            expand(batch_size, self.beam_size * size_multiplier, *last_dims)
                except RuntimeError:
                    pdb.set_trace()

                # shape: (batch_size * beam_size, *)
                try:
                    copy_state[key] = state_tensor.\
                        reshape(batch_size, self.beam_size * size_multiplier, *last_dims).\
                        gather(1, expanded_backpointer).\
                        reshape(batch_size * self.beam_size * size_multiplier, *last_dims)
                except RuntimeError:
                    pdb.set_trace()
            # Keep only the pieces of the auxiliaries corresponding to the
            # ancestors created this iteration.
            for key, aux in auxiliaries.items():
                new_aux = []
                aux = [x for i in range(n_nonconfident+1) for x in aux]
                for ith, indices in enumerate(backpointer.tolist()):
                    new_aux += [aux[ith * self.beam_size * size_multiplier + index].copy() for index in indices]
                auxiliaries[key] = new_aux

            tracked_state = copy_state[tracked_state_name]
            _, *last_dims = tracked_state.size()
            # shape: [(batch_size, beam_size, *)]
            tracked_states.append(tracked_state.reshape(batch_size, self.beam_size * size_multiplier, *last_dims))

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]
        # shape: [(batch_size, beam_size, 1, *)]
        reconstructed_tracked_states = [tracked_states[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: [(batch_size, beam_size, 1, *)]
            _, _, *last_dims = tracked_states[timestep].size()
            expanded_cur_backpointers = cur_backpointers.\
                view(batch_size, self.beam_size * size_multiplier, *([1] * len(last_dims))).\
                expand(batch_size, self.beam_size * size_multiplier, *last_dims)
            cur_tracked_state = tracked_states[timestep].gather(1, expanded_cur_backpointers).unsqueeze(2)

            reconstructed_tracked_states.append(cur_tracked_state)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        # shape: [(batch_size, beam_size, 1, *)]
        _, _, *last_dims = tracked_states[0].size()
        expanded_cur_backpointers = cur_backpointers.\
            view(batch_size, self.beam_size * size_multiplier, *([1] * len(last_dims))).\
            expand(batch_size, self.beam_size * size_multiplier, *last_dims)
        final_tracked_state = tracked_states[0].gather(1, expanded_cur_backpointers).unsqueeze(2)

        reconstructed_tracked_states.append(final_tracked_state)

        # shape: (batch_size, beam_size, max_steps, *)
        all_tracked_states = torch.cat(list(reversed(reconstructed_tracked_states)), 2)

        # shape: (batch_size * beam_size)
        tracked_auxiliary = auxiliaries[tracked_auxiliary_name]
        # shape: (batch_size, beam_size)
        for beam_index in range(self.beam_size * size_multiplier):
            for i in range(beam_index, len(tracked_auxiliary), self.beam_size):
                tracked_auxiliaries[beam_index % size_multiplier][i // (self.beam_size * size_multiplier)] = tracked_auxiliary[i % size_multiplier]

        return all_predictions, all_tracked_states, last_log_probabilities, tracked_auxiliaries, size_multiplier
