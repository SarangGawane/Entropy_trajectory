"""Generate per-token entropy/uncertainty traces, in the schema of
Shichengf/entropy-acc-traces.

NOTE ON FIDELITY: the published dataset came from the author's own SFT
checkpoints (`qwen3-8b-instruct-sft`, `qwen3-8b-base-sft`), which were never
released, and no generation code was published either. This script reconstructs
the pipeline against a public model. Absolute entropy values will NOT match the
published traces -- different weights, different distributions. Only the schema
and the signal definitions are shared.

Decoding is a manual loop rather than model.generate(): HF merges custom
LogitsProcessors with the temperature/top-p warpers, which makes it ambiguous
whether a processor sees raw or already-tempered logits. Since entropy is the
measurement, that ambiguity is not acceptable -- here, signals are always read
off the raw logits and sampling is applied afterwards.

Usage:
  python generate_traces.py --bench aime24  --out out/aime24.jsonl.zst
  python generate_traces.py --bench math500 --out out/math500.jsonl.zst
"""

import argparse
import json
import os
import re
import time

import torch
import zstandard as zstd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import PER_STEP_KEYS, TOPK, step_signals

BENCHES = {
    "aime24":  dict(path="HuggingFaceH4/aime_2024", split="train", qcol="problem", acol="answer"),
    "math500": dict(path="HuggingFaceH4/MATH-500", split="test",  qcol="problem", acol="answer"),
}


# ---------------------------------------------------------------- answers

BOXED = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str):
    """Return the content of the LAST \\boxed{...}, brace-matched."""
    starts = [m.end() for m in BOXED.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, out = 1, []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if not depth:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() if not depth else None


def norm_answer(s):
    """Loose numeric/string normalisation -- enough for AIME (integers) and
    most of MATH-500. Not a full CAS equivalence check."""
    if s is None:
        return None
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"^\\left\(|\\right\)$", "", s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        s = str(int(float(s)))
    return s


def is_correct(pred, gold):
    p, g = norm_answer(pred), norm_answer(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


# ---------------------------------------------------------------- decoding

COMPACT_EVERY = 8


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens, temperature, top_p, top_k, device):
    """Manual batched decode. Returns one dict per prompt with text, tokens and
    the per_step signal block, truncated at that sequence's EOS.

    Three things here are deliberate, and all three are about speed without
    touching the measurement:

    1. Nothing is read back to the host inside the loop. Reading `nxt[b]` per
       sequence per step forces a full device sync on every token, which is what
       dominated step time before -- signals and tokens go into preallocated GPU
       buffers and cross to the host once, at the end.
    2. A sequence that emits EOS is dropped from the batch AND from the KV cache
       (`batch_select_indices`), so a ragged batch costs the slowest sequence's
       length at a *shrinking* width rather than a fixed one. Buffers are indexed
       by original row, so compaction never renumbers anything.
    3. Compaction is checked every COMPACT_EVERY steps rather than every step:
       the check itself needs a sync, and syncing per step stops the CPU from
       queueing the next step's kernels while the GPU works. Between checks, a
       dead row is carried with attention_mask 0 exactly as before -- it writes
       past its own recorded length, and those rows are sliced off at the end.

    Signals are still read off the raw logits before sampling, as in metrics.py.
    """
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
    input_ids, attn = enc.input_ids, enc.attention_mask
    B = input_ids.shape[0]

    eos_ids = set(tok.all_special_ids) | ({tok.eos_token_id} if tok.eos_token_id is not None else set())
    eos_t = torch.tensor(sorted(i for i in eos_ids if i is not None), device=device)

    # Buffers are (T, B_orig, ...) and indexed by ORIGINAL row.
    bufs = {}
    for k in PER_STEP_KEYS:
        if k == "top20_token_ids":
            bufs[k] = torch.zeros((max_new_tokens, B, TOPK), dtype=torch.long, device=device)
        elif k == "top20_logprobs":
            bufs[k] = torch.zeros((max_new_tokens, B, TOPK), dtype=torch.float32, device=device)
        else:
            bufs[k] = torch.zeros((max_new_tokens, B), dtype=torch.float32, device=device)
    tok_buf = torch.zeros((max_new_tokens, B), dtype=torch.long, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)

    alive = torch.arange(B, device=device)                  # original row of each batch slot
    live = torch.ones(B, dtype=torch.bool, device=device)   # not-yet-EOS, over current slots
    past, cur = None, input_ids

    for t in range(max_new_tokens):
        out = model(input_ids=cur, attention_mask=attn, past_key_values=past,
                    use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        logits = out.logits[:, -1, :]                      # (B, V) RAW logits

        sig = step_signals(logits)                          # signals BEFORE sampling

        # --- sampling (does not affect recorded signals) ---
        z = logits.float()
        if temperature and temperature > 0:
            z = z / temperature
            if top_k:
                kth = z.topk(top_k, dim=-1).values[:, -1:]
                z = z.masked_fill(z < kth, float("-inf"))
            if top_p and top_p < 1.0:
                sz, si = z.sort(dim=-1, descending=True)
                cum = sz.softmax(-1).cumsum(-1)
                drop = cum - sz.softmax(-1) > top_p
                sz = sz.masked_fill(drop, float("-inf"))
                z = torch.full_like(z, float("-inf")).scatter(-1, si, sz)
            nxt = torch.multinomial(z.softmax(-1), 1).squeeze(-1)
        else:
            nxt = z.argmax(-1)

        # Record, then mark EOS: the EOS token itself is emitted and measured.
        for k in PER_STEP_KEYS:
            bufs[k][t].index_copy_(0, alive, sig[k])
        tok_buf[t].index_copy_(0, alive, nxt)
        lengths.index_add_(0, alive, live.long())

        live &= ~torch.isin(nxt, eos_t)

        cur = nxt.unsqueeze(-1)
        attn = torch.cat([attn, live.long().unsqueeze(-1)], dim=-1)

        if (t + 1) % COMPACT_EVERY == 0:
            n_live = int(live.sum())                        # the loop's only sync
            if n_live == 0:
                break
            if n_live < live.shape[0]:
                idx = live.nonzero(as_tuple=True)[0]
                alive, live, attn, cur = alive[idx], live[idx], attn[idx], cur[idx]
                past.batch_select_indices(idx)

    lens = lengths.tolist()
    host = {k: bufs[k].cpu() for k in PER_STEP_KEYS}
    tok_host = tok_buf.cpu()

    recs = []
    for b in range(B):
        L = lens[b]
        gen = tok_host[:L, b].tolist()
        per_step = {}
        for k in PER_STEP_KEYS:
            v = host[k][:L, b]
            if L == 0:
                per_step[k] = []
            elif k == "top20_token_ids":
                per_step[k] = v.tolist()
            elif v.ndim == 1:
                # 8 dp: p_tail is often ~1e-7 on confident steps and 6 dp would
                # flush it to zero.
                per_step[k] = [round(float(x), 8) for x in v.tolist()]
            else:
                per_step[k] = [[round(float(x), 6) for x in row] for row in v.tolist()]
        recs.append({
            "tokens": gen,
            "text": tok.decode(gen, skip_special_tokens=True),
            "num_steps": L,
            "per_step": per_step,
        })
    return recs


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=BENCHES, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=1, help="samples per problem")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0,
                    help="resume: skip problems with dataset idx < START. The recorded "
                         "idx stays the true dataset index, so output merges with an "
                         "earlier partial run.")
    ap.add_argument("--sort-key", default=None,
                    help="dataset column to order problems by before batching (e.g. "
                         "'level'). A batch costs its SLOWEST member, so grouping "
                         "similar-difficulty problems cuts wasted decode steps. Does "
                         "not affect the recorded idx.")
    ap.add_argument("--temperature", type=float, default=0.6)   # Qwen3 thinking-mode defaults
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    cfg = BENCHES[args.bench]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    ).eval()

    ds = load_dataset(cfg["path"], split=cfg["split"])
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    items = []
    for i, row in enumerate(ds):
        msgs = [{"role": "user", "content": row[cfg["qcol"]]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if i < args.start:
            continue
        for r in range(args.rollouts):
            items.append(dict(idx=i, rollout=r, prompt=prompt,
                              problem=row[cfg["qcol"]], gold=str(row[cfg["acol"]]),
                              sort=row.get(args.sort_key) if args.sort_key else None))
    if args.start:
        print(f"resuming at idx {args.start}: {len(items)} of {len(ds)} problems remain", flush=True)
    if args.sort_key:
        if any(it["sort"] is None for it in items):
            raise SystemExit(f"--sort-key {args.sort_key!r} not in columns {ds.column_names}")
        items.sort(key=lambda it: it["sort"])
        print(f"batching in order of '{args.sort_key}' to shrink ragged-batch waste", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0, n_done, n_ok = time.time(), 0, 0

    with open(args.out, "wb") as fh:
        with zstd.ZstdCompressor(level=3).stream_writer(fh) as w:
            for s in range(0, len(items), args.batch_size):
                chunk = items[s:s + args.batch_size]
                recs = generate_batch(
                    model, tok, [c["prompt"] for c in chunk], args.max_new_tokens,
                    args.temperature, args.top_p, args.top_k, device,
                )
                for c, rec in zip(chunk, recs):
                    pred = extract_boxed(rec["text"])
                    ok = is_correct(pred, c["gold"])
                    n_ok += ok
                    n_done += 1
                    rec.update(
                        bench=args.bench, model=args.model, idx=c["idx"], rollout=c["rollout"],
                        problem=c["problem"], gold_answer=c["gold"], pred_answer=pred,
                        correct=bool(ok),
                        sampling=dict(temperature=args.temperature, top_p=args.top_p,
                                      top_k=args.top_k, seed=args.seed,
                                      max_new_tokens=args.max_new_tokens),
                    )
                    w.write((json.dumps(rec) + "\n").encode())
                el = time.time() - t0
                print(f"[{n_done}/{len(items)}] acc={n_ok/n_done:.3f} "
                      f"steps={recs[-1]['num_steps']} elapsed={el/60:.1f}m "
                      f"eta={(el/n_done)*(len(items)-n_done)/60:.1f}m", flush=True)

    print(f"DONE {args.out} n={n_done} acc={n_ok/max(n_done,1):.4f}")


if __name__ == "__main__":
    main()
