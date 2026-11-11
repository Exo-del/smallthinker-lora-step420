# SmallThinker-4BA0.6B-Instruct — thinking + tool-calling LoRA (step 429)

LoRA adapter that teaches the base model a reasoning + tool-calling format, and
lifts its coding behaviour, from a 19.3 M-token supervised mix
(49 % tool use / 30 % reasoning-STEM / 21 % coding).

**This adapter** or its parent [`smallthinker-lora-v2`](https://github.com/Exo-del/smallthinker-lora-v2)
— step 429 is the recommended download (slightly better fit, tighter format).

## What it produces

```
<think>
The user wants me to read the config.yaml file and get the database host.
I'll start by reading the file to see what's in it.
</think>
<tool_call>
{"name": "read_file", "arguments": {"path": "config.yaml"}}
</tool_call><|im_end|>
```

The base model has **neither** thinking nor tool calling. Measured on 7 prompts
(5 synthetic + 2 real corpus), greedy decoding:

| | base | this adapter |
|---|---|---|
| p(`<think>`) at turn start | 0 · rank 77,968 | **0.997–0.998 · rank 1** |
| p(`</think>`) after reasoning | — | **0.998–0.9999 · rank 1** |
| opens with `<think>` | 0/7 | **7/7** |
| closes with `</think>` | — | **7/7** |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Tiiny/SmallThinker-4BA0.6B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16,
    device_map="auto")
model = PeftModel.from_pretrained(model, "Exo-del/smallthinker-lora-step420")

tools = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a file from the working directory.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]

msgs = [{"role": "system", "content": "You are a precise software engineering assistant."},
        {"role": "user", "content": "Read config.yaml and tell me the database host."}]

prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                 add_generation_prompt=True)
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                     max_new_tokens=256, do_sample=False,
                     eos_token_id=[151643, 151645])
print(tok.decode(out[0][tok(prompt)["input_ids"].__len__():], skip_special_tokens=False))
```

The format is **inline**, not a chat-template special case: put your prompt
through `apply_chat_template` with `add_generation_prompt=True` and the adapter
writes `<think>`, reasoning, `</think>`, then either prose or `<tool_call>`.

`pad_token_id` is `<|endoftext|>` (151643) and `<|im_end|>` (151645) ends a turn.

## Training

| | |
|---|---|
| adapter | r=8, alpha=16, dropout 0.05, targets `q,k,v,o` + `lm_head` |
| run 1 (`v1`) | 1,737 steps, 10 h — tool calls worked, `<think>` never emitted |
| run 2..4 | tag-weighted loss + `lm_head` LoRA + symmetric tag mass, ~1.2 h |
| this adapter | +429 steps, 2.5 h — 8.8 % of an epoch, format verified after |
| hardware | 2×T4, seq 3072, peak 12.2 GiB, manual activation checkpointing |

The whole story of why `<think>` needed a weighted loss is in
[`NOTES.md`](NOTES.md); the short version is that the tags are 0.4 % of the
supervised tokens, so a mean loss leaves them untrained, and a frozen `lm_head`
cannot move an output row it never emits.

## Caveat

This is a **format-and-style** adapter over a 4B-MoE base, trained for ~0.1 epoch
total. It reliably produces the structure and the reasoning style; it is not a
frontier coding model and has not been evaluated on benchmarks.
