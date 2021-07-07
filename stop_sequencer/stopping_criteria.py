from typing import List

import torch
from transformers import AutoTokenizer
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


tokenizer = AutoTokenizer.from_pretrained("gpt2")


class StopSequenceCriteria(StoppingCriteria):

    def __init__(
        self,
        model_type,
        tokenizer,
        stop_texts: List[str],
        input_length,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert model_type.lower() in ["causal", "seq2seq"], \
            "model type must be one of [causal, seq2seq]"

        self.model_type = model_type
        self.stop_texts = stop_texts
        self.tokenizer = tokenizer
        self.input_length = input_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:

        input_ids = input_ids.long().tolist()

        if self.model_type == "causal":
            new_input_ids = [i[self.input_length:] for i in input_ids]
        else:
            new_input_ids = input_ids

        stops = []
        for text in self.stop_texts:
            stop = []
            for input_id in new_input_ids:
                decoded = tokenizer.decode(input_id)
                stop.append(text in decoded)
            stops.append(all(stop))

        return any(stops)
