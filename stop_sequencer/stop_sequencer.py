from copy import deepcopy
from typing import Optional, List

import torch

from stop_sequencer.stopping_criteria import StopSequenceCriteria


class StopSequencer(object):

    def __init__(self, model, model_type, tokenizer):
        assert model_type.lower() in ["causal", "seq2seq"], \
            "model type must be one of [causal, seq2seq]"

        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.orig_get_stopping_criteria = model._get_stopping_criteria
        self.orig_generate = model.generate
        self.options = {}

    def register_stop_texts(self, stop_texts, input_length: int):

        def _get_stopping_criteria(
            max_length: Optional[int],
            max_time: Optional[float],
            max_new_tokens: Optional[int],
            start_length: int,
        ):
            stopping_criterias = self.orig_get_stopping_criteria(
                max_length,
                max_time,
                max_new_tokens,
                start_length,
            )
            stopping_criterias_ = deepcopy(stopping_criterias)

            criteria = StopSequenceCriteria(
                input_length=input_length,
                model_type=self.model_type,
                stop_texts=stop_texts,
                tokenizer=self.tokenizer,
            )

            stopping_criterias_.append(criteria)

            return stopping_criterias_

        self.model._get_stopping_criteria = _get_stopping_criteria

        return self.model
