"""Test the uncertainty-vs-correctness question on the generated traces.

The whole point of the trace dataset is: do the model's own uncertainty signals
separate correct from incorrect reasoning? Two things make that easy to get wrong,
and both are handled here rather than left to the reader:

  1. TRUNCATION. A trace that hits the token cap never emits a \\boxed{} and is
     scored "incorrect" -- but it is unfinished work, not a wrong answer. Including
     those manufactures a large fake entropy gap, because high entropy -> longer
     rambling -> hits the cap. Every test below runs on naturally-terminated traces
     ONLY, and the truncated ones are reported separately.
  2. LENGTH. Mean entropy correlates with trace length, so a signal difference can
     be a length difference wearing a hat. Reported here as a partial correlation
     holding length fixed, plus a within-length-band breakdown.

All seven scalar signals are tested, so p-values get a Holm correction -- testing
until something clears 0.05 is how noise gets published.

Usage:  python analyze_entropy.py [--bench math500|aime24]
"""

import argparse
import io
import json
import math
import statistics as st

import zstandard as zstd

SCALARS = ["entropy", "varentropy", "top1_prob", "top2_prob", "margin",
           "self_certainty", "p_tail"]
FILES = {"math500": "out/math500.jsonl.zst", "aime24": "out/aime24.jsonl.zst"}


def read(path):
    recs = []
    with open(path, "rb") as fh:
        buf = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh),
                               encoding="utf-8", errors="replace")
        while True:
            try:
                line = buf.readline()
            except Exception:
                break
            if not line:
                break
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: torn record after idx {recs[-1]['idx']}; "
                      f"analysing the {len(recs)} complete ones")
                break
    return recs


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return float("nan")
    return sum((a[i] - ma) * (b[i] - mb) for i in range(len(a))) / math.sqrt(va * vb)


def mwu(x, y):
    """Mann-Whitney U, normal approximation with tie correction. Two-sided.
    Non-parametric on purpose: per-trace signal means are not normal."""
    merged = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    n = len(merged)
    R = [0.0] * n
    i = 0
    ties = 0
    while i < n:
        j = i
        while j + 1 < n and merged[j + 1][0] == merged[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            R[k] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    n1, n2 = len(x), len(y)
    R1 = sum(R[k] for k in range(n) if merged[k][1] == 0)
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt((n1 * n2 / 12) * ((n1 + n2 + 1) - ties / ((n1 + n2) * (n1 + n2 - 1))))
    if sd == 0:
        return float("nan"), float("nan")
    z = (U1 - mu) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p, U1 / (n1 * n2)          # p, common-language effect size (AUC)


def holm(pvals):
    """Holm-Bonferroni: control family-wise error across the signals tested."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (len(pvals) - rank) * pvals[i])
        prev = adj[i] = max(prev, v)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=FILES, default="math500")
    args = ap.parse_args()

    recs = read(FILES[args.bench])
    cap = recs[0]["sampling"]["max_new_tokens"]
    for r in recs:
        r["mH"] = {k: mean(r["per_step"][k]) for k in SCALARS}
        r["trunc"] = r["num_steps"] >= cap or r["pred_answer"] is None

    term = [r for r in recs if not r["trunc"]]
    tc = [r for r in term if r["correct"]]
    ti = [r for r in term if not r["correct"]]

    print(f"=== {args.bench}  (n={len(recs)}, cap={cap}) ===")
    print(f"  raw accuracy (truncated counted wrong) : {mean([r['correct'] for r in recs]):.4f}")
    print(f"  truncated / no answer                  : {len(recs)-len(term)} "
          f"({(len(recs)-len(term))/len(recs):.1%})  <- excluded below, unfinished not wrong")
    print(f"  naturally terminated                   : {len(term)}"
          f"  -> {len(tc)} correct, {len(ti)} genuine errors")
    if term:
        print(f"  accuracy among terminated              : {len(tc)/len(term):.4f}")
    if len(ti) < 5 or len(tc) < 5:
        print(f"\n  too few genuine errors (n={len(ti)}) to support any claim. stop.")
        return

    print(f"\n--- signal vs correctness, TERMINATED ONLY (n={len(term)}) ---")
    print(f"  {'signal':<15}{'correct':>10}{'incorrect':>11}{'gap':>10}{'AUC':>7}"
          f"{'p':>9}{'p_holm':>9}")
    ps, rows = [], []
    for k in SCALARS:
        a = [r["mH"][k] for r in tc]
        b = [r["mH"][k] for r in ti]
        p, auc = mwu(a, b)
        ps.append(p)
        rows.append((k, mean(a), mean(b), mean(b) - mean(a), auc, p))
    for (k, ma, mb, gap, auc, p), pa in zip(rows, holm(ps)):
        star = "  *" if pa < 0.05 else ""
        print(f"  {k:<15}{ma:>10.4f}{mb:>11.4f}{gap:>+10.4f}{auc:>7.3f}{p:>9.4f}{pa:>9.4f}{star}")
    print("  (gap = incorrect - correct; AUC = P(random correct trace scores higher);")
    print("   * = survives Holm correction across the 7 signals)")

    print(f"\n--- is it just length? ---")
    L = [float(r["num_steps"]) for r in term]
    C = [float(r["correct"]) for r in term]
    print(f"  mean length: correct={mean([r['num_steps'] for r in tc]):.0f}  "
          f"incorrect={mean([r['num_steps'] for r in ti]):.0f}")
    print(f"  corr(correct, length) = {pearson(C, L):+.3f}")
    print(f"  {'signal':<15}{'r(correct,sig)':>16}{'r(sig,length)':>15}{'partial|length':>16}")
    for k in SCALARS:
        S = [r["mH"][k] for r in term]
        r_cs, r_cl, r_sl = pearson(C, S), pearson(C, L), pearson(S, L)
        denom = math.sqrt((1 - r_cl ** 2) * (1 - r_sl ** 2))
        pc = (r_cs - r_cl * r_sl) / denom if denom else float("nan")
        print(f"  {k:<15}{r_cs:>+16.3f}{r_sl:>+15.3f}{pc:>+16.3f}")

    print(f"\n--- entropy gap within length bands (tertiles) ---")
    srt = sorted(term, key=lambda r: r["num_steps"])
    t = len(srt) // 3
    for name, band in [("short", srt[:t]), ("mid", srt[t:2 * t]), ("long", srt[2 * t:])]:
        bc = [r["mH"]["entropy"] for r in band if r["correct"]]
        bi = [r["mH"]["entropy"] for r in band if not r["correct"]]
        rng = f"{band[0]['num_steps']}-{band[-1]['num_steps']}tok"
        if len(bi) >= 3 and len(bc) >= 3:
            p, _ = mwu(bc, bi)
            print(f"  {name:<6}({rng:>14}): correct={mean(bc):.4f}(n={len(bc):>3})  "
                  f"incorrect={mean(bi):.4f}(n={len(bi):>3})  gap={mean(bi)-mean(bc):+.4f}  p={p:.3f}")
        else:
            print(f"  {name:<6}({rng:>14}): too few errors (n={len(bi)})")


if __name__ == "__main__":
    main()
