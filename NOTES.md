# SmallThinker LoRA — step 420 (durability checkpoint)

Continuation of `Exo-del/smallthinker-lora-v2` for **429 optimizer steps** (2.5 h,
2×T4, 3,432 examples ≈ 8.8 % of an epoch) with the format weights left active.

Purpose was a durability test, not a capability run: does the think format
survive real corpus gradient? v2's format lived in a rank-8 `lm_head` delta
trained for only 200 steps, and a long run applies thousands of steps of pressure
in many other directions.

| setting | value |
|---|---|
| steps | 429, 2.5 h, 2×T4, peak 12.22 GiB, `skipped_oom 0` |
| lr | 8e-5, cosine over 900 steps, warmup 30 → 4.65e-5 at the end |
| loss | `--tag-weight 200 --tag-ids 151668 --first-target-k 1 --first-target-weight 200` |
| LoRA | `q,k,v,o` + `lm_head`, r=8, alpha=16 |
| init | `Exo-del/smallthinker-lora-v2` |

## Training evidence

`tag_ce` — the teacher-forced loss on the structural tags — stayed at
**~0.0001** through the whole run (step 400: 0.0001, 410: 0.0002, 420: 0.0001),
final `last_loss` 0.242. On corpus data the model keeps predicting `<think>` and
`</think>` at essentially probability 1 while learning ordinary tokens.

Generation-level verification (format probe + before/after CE on identical
examples) runs in a separate probe-only notebook, because the post-training probe
in this run hit a `NameError` in `format_probe.py` (`extract_reasoning` was called
but not defined) — the training itself had already exited 0.

## Files

`adapter_model.safetensors` — LoRA only (the 466 MB `lm_head.base_layer.weight`
PEFT also dumps is stripped) · `adapter_config.json` (base path = HF id) ·
`run_summary.json`.
