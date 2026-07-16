# Entropy trajectories for reasoning traces (Qwen3-8B)

Per-token uncertainty traces for math reasoning, in the schema of
[`Shichengf/entropy-acc-traces`](https://huggingface.co/datasets/Shichengf/entropy-acc-traces),
generated against **public** Qwen3-8B weights.

**530 traces / 2,645,155 decode steps**, each step carrying 9 uncertainty signals
read off the raw next-token logits.

| bench | traces | cap | raw acc | truncated | analyzable | genuine errors |
|---|---|---|---|---|---|---|
| MATH-500 | 500 | 8,192 | 0.670 | 99 (19.8%) | 401 | 74 |
| AIME24 | 30 | 16,384 | 0.667 | 8 (26.7%) | 22 | 2 |

## Headline result: no signal predicts correctness

None of the seven scalar signals separates correct from incorrect reasoning at the
trace-mean level. Every AUC lands in 0.46–0.54 (0.5 = chance) and every
Holm-corrected p-value is 1.000.

```
signal            correct  incorrect       gap    AUC        p   p_holm
entropy            0.2443     0.2386   -0.0057  0.532   0.3827   1.0000
varentropy         0.2460     0.2423   -0.0038  0.524   0.5116   1.0000
top1_prob          0.9068     0.9092   +0.0024  0.465   0.3463   1.0000
margin             0.8411     0.8451   +0.0040  0.462   0.3016   1.0000
self_certainty    33.9149    34.1209   +0.2060  0.476   0.5123   1.0000
```

Reproduce with `python analyze_entropy.py --bench math500`.

### Two traps this repo exists to document

**1. Truncation masquerades as error.** A trace that hits the token cap never emits
`\boxed{}` and scores "incorrect" — but it is unfinished, not wrong. At AIME24's
16k cap, 8 of 10 "errors" were truncations; only 2 were real. Including them
manufactures a large fake entropy gap, because high entropy → longer rambling →
hits the cap → no answer. Mean entropy correlates +0.373 with trace length on that
run. **Always filter to naturally-terminated traces before any correctness claim** —
`analyze_entropy.py` does this by default and reports the truncated count separately.

The source dataset's own filename (`aime24_stage2_mt16384.jsonl.zst`) encodes the
same 16k cap as a separate stage-2 file, so the published traces likely contain the
same truncated rollouts and need the same filtering.

**2. Small-n effects evaporate.** On the first 135 MATH-500 problems (21 genuine
errors), incorrect traces looked *more confident* than correct ones — gap −0.0335,
p=0.036, and the length confound was genuinely absent. It was noise. With all 500
(74 errors) the gap fell 6x to −0.0057 at p=0.383. Across this project the same
question returned "higher entropy" (artifact), then "lower" (noise), then "neither".

The null is specific: it rules out **trace-mean** signals predicting correctness for
stock Qwen3-8B on MATH-500 at one rollout/problem. Averaging over ~3,700 steps is
blunt and may wash out a localized signal — peak entropy, entropy at specific
reasoning steps, or trajectory *shape* remain open, and all 2.6 M per-step values
are in the data.

## Files

| file | what |
|---|---|
| `metrics.py` | the 9 signals, computed from raw logits (temperature 1.0) |
| `generate_traces.py` | manual batched decode + trace generation |
| `analyze_entropy.py` | correctness analysis with truncation + length controls |
| `merge_math500.py` | merges a partial run with a `--start` resume |
| `read_traces.py`, `download.py` | dataset/model fetch helpers |

Signals are read off **raw** logits before sampling. Entropy of a top-p-filtered
distribution measures the sampler, not the model, so sampling params affect which
token is drawn and never the recorded signals. This is also why decoding is a manual
loop rather than `model.generate()`: HF merges custom `LogitsProcessor`s with the
temperature/top-p warpers, leaving it ambiguous whether a processor sees raw or
already-tempered logits.

## Data

`out/aime24.jsonl.zst` ships whole (43 MB). `out/math500.jsonl.zst` is 262 MB —
past GitHub's 100 MB per-file limit — so it is split into `out/parts/`. Reassemble:

```bash
bash out/parts/reassemble.sh     # verifies sha256
```

One JSON record per trace: `idx`, `problem`, `gold_answer`, `pred_answer`,
`correct`, `num_steps`, `tokens`, `text`, `sampling`, and `per_step` — the 9
signals as arrays of length `num_steps`.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python generate_traces.py --bench math500 --batch-size 6 --max-new-tokens 8192 \
    --sort-key level --out out/math500.jsonl.zst
```

~5.6 h on one RTX 4090. Notes on the knobs, all measured:

- **`--batch-size 6` is the ceiling on a 24 GB card.** Decode is bandwidth-bound
  reading 16.4 GB of weights every step, so step time is nearly flat in batch width
  (28.5 ms at batch 6 vs batch 4) — width is free, and batch size is the only real
  lever. But 20% of traces run to the cap, so peak KV is `batch x 8192 x 0.1406 MiB`:
  batch 6 → 22.0 GiB, batch 8 → 24.3 GiB → OOM.
- **`--sort-key level`** buys ~8%. A batch costs its *slowest* member, and with 20%
  hitting the cap, ~74% of batches drag to 8192 while most rows finish near 3,300.
  `level` correlates +0.472 with length; an oracle length-sort would save 40%.
- **`--start N`** resumes after an interrupted run; recorded `idx` stays the true
  dataset index so halves merge. RNG stream differs from an unbroken run, so
  resumed traces are fresh samples, not byte-identical reproductions.
- CPU offload is not viable despite 192 cores: PyTorch's CPU bf16 decode hits
  ~27 GB/s effective (~6% of theoretical), giving 1.7 tok/s vs the GPU's ~200, and
  it gets *slower* with batching.

## Caveats

- **This does not replicate the published traces.** They came from the authors'
  `qwen3-8b-instruct-sft` / `qwen3-8b-base-sft` checkpoints, which were never
  released, with no generation code published. Absolute entropy values will not
  match — different weights, different distributions. Only the schema and signal
  definitions are shared.
- One rollout per problem. For statistical power on correctness, multiple rollouts
  per problem would matter more than a larger cap.
- `norm_answer` is loose numeric/string normalisation, not a CAS equivalence check.
  Fine for AIME (integers) and most of MATH-500; some correct-but-oddly-formatted
  answers will score wrong.
