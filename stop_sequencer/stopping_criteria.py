import torch

from typing import List
from transformers import StoppingCriteria

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
                decoded = self.tokenizer.decode(input_id)
                stop.append(text in decoded)
            stops.append(all(stop))

        return any(stops)
