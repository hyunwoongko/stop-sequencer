from copy import deepcopy
from typing import Optional, List
from stop_sequence.stopping_criteria import StopSequenceCriteria


class StopSeqeuencer(object):

    def __init__(self, model):
        self.model = model
        self.orig_get_stopping_criteria = model._get_stopping_criteria

    def register_stop_tokens(self, tokens: List[List[int]]):

        def _get_stopping_criteria(
            model,
            max_length: Optional[int],
            max_time: Optional[float],
            max_new_tokens: Optional[int],
            start_length: int,
        ):
            stopping_criterias = self.orig_get_stopping_criteria(
                model,
                max_length,
                max_time,
                max_new_tokens,
                start_length,
            )
            stopping_criterias_ = deepcopy(stopping_criterias)

            stopping_criterias_.append(
                StopSequenceCriteria(stop_sequence_tokens=tokens))

            return stopping_criterias_

        self.model._get_stopping_criteria = _get_stopping_criteria

        return self.model
