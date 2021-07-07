from typing import List

import torch
from transformers import StoppingCriteria


def is_subset(l, s):
    subset = False

    if len(s) == 0:
        return False
    elif s == l:
        return True
    elif len(s) > len(l):
        return False
    else:
        for i in range(len(l)):
            if l[i] == s[0]:
                n = 1
                while (n < len(s)) and (l[i + n] == s[n]):
                    n += 1

                if n == len(s):
                    subset = True

    return subset


class StopSequenceCriteria(StoppingCriteria):

    def __init__(self, stop_sequence_tokens: List[List[int]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_sequence_tokens = stop_sequence_tokens

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:

        input_ids = input_ids.long().tolist()

        for tokens in self.stop_sequence_tokens:
            if is_subset(input_ids, tokens):
                return True

        return False
