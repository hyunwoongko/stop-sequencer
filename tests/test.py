from transformers import AutoModelForCausalLM, AutoTokenizer
from stop_sequencer import StopSequencer

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

print("w/o stop-sequencer:")
print(tokenizer.batch_decode(outputs[:, tokens.size(-1):])[0])
print()
# ive been watching TV for a long time. Ryan: I have been watching TV since I was 12 years old. Kevin: So what do you want me to do? Ryan: Well, I want you to watch TV. You know what I mean? I'm going to be watching TV. I'm not going to sit down and watch TV. I don't want to

stop_sequencer = StopSequencer(
    model,
    model_type="causal",
    tokenizer=tokenizer,
)

stop_texts = ["Ryan:", "Kevin:"]
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

print("w/ stop-sequencer:")
outputs = tokenizer.batch_decode(outputs[:, tokens.size(-1):])[0]
print(outputs)
print()
# ive been watching TV for a long time. Ryan: I have<|endoftext|>

print("w/ stop-sequencer + post-process:")
for s in stop_texts:
    outputs = outputs.split(s)[0].strip()
print(outputs)
print()
# ive been watching TV for a long time.
