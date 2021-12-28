# Stop Sequencer
- Implementation of stop sequencer for Huggingface Transformers.
- Note post-processing must be used together because limitation of transformers implementation.
<br><br>
  
## 1. Installation
```console
pip install stop-sequencer
```
<br>

## 2. Usage
### 2.1. Generation without StopSequencer
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer(
    "Kevin: Hello "
    "Ryan: Hi "
    "Kevin: What are you doing? "
    "Ryan: I am watching TV. you? "
    "Kevin: ",
    return_tensors="pt",
)["input_ids"]

outputs = model.generate(
    tokens,
    num_beams=5,
    no_repeat_ngram_size=4,
    repetition_penalty=1.5,
    max_length=100,
)

outputs = tokenizer.batch_decode(outputs[:, tokens.size(-1):], skip_special_tokens=True)[0]
print(outputs)
```
```
ive been watching TV for a long time. Ryan: I have been watching TV since I was 12 years old. Kevin: So what do you want me to do? Ryan: Well, I want you to watch TV. You know what I mean? I'm going to be watching TV. I'm not going to sit down and watch TV. I don't want to
```
<br><br>

### 2.2. Generation with StopSequencer

```python
from stop_sequencer import StopSequencer

stop_texts = ["Ryan:", "Kevin:"]

stop_sequencer = StopSequencer(
    model,
    model_type="causal",  # or seq2seq
    tokenizer=tokenizer,
)

model = stop_sequencer.register_stop_texts(
    stop_texts=stop_texts,
    input_length=tokens.size(-1),
)

outputs = model.generate(
    tokens,
    num_beams=5,
    no_repeat_ngram_size=4,
    repetition_penalty=1.5,
    max_length=100,
)

outputs = tokenizer.batch_decode(outputs[:, tokens.size(-1):], skip_special_tokens=True)[0]
print(outputs)
```
```
ive been watching TV for a long time. Ryan: I have
```
You can see that `Ryan: I have` is contained in the generation result and then generation is finished. The generation can be terminated after stop text (`Ryan: I have`) is generated because of the limitation of Huggingface Transformers.
<br><br>

### 3. Generation with StopSequencer + post-processing
Therefore, post-processing must be performed to completely exclude stop texts from generated text.
```python
for s in stop_texts:
    outputs = outputs.split(s)[0].strip()
    
print(outputs)
```
```
ive been watching TV for a long time.
```
<br><br>

## License
```
Copyright 2021 Hyunwoong Ko.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
