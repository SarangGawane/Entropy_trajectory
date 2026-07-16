"""Merge the salvaged MATH-500 partial with the resumed run into one file.

The first run was killed mid-write at problem 135, leaving out/math500.jsonl.zst
with 135 good records followed by a half-written one (the zstd frame was never
closed). math500_partial.jsonl.zst is those 135 records rewritten into a clean
frame; math500_resume.jsonl.zst carries idx 135-499 from the --start resume.

Both halves record the true dataset idx, so they concatenate and sort. The
resume was decoded in level order, so its file is not in idx order -- that is
what the sort here is for.

Usage:  python merge_math500.py [--force]
"""

import argparse
import io
import json
import os
import sys

import zstandard as zstd

PARTIAL = "out/math500_partial.jsonl.zst"
RESUME = "out/math500_resume.jsonl.zst"
MERGED = "out/math500.jsonl.zst"
N_EXPECTED = 500


def read(path):
    """Read records, stopping cleanly at a torn tail rather than raising."""
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
                print(f"  {path}: stopped at torn record after idx {recs[-1]['idx']}")
                break
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="write even if fewer than 500 problems are present")
    args = ap.parse_args()

    recs = []
    for p in (PARTIAL, RESUME):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        r = read(p)
        print(f"  {p}: {len(r)} records")
        recs.extend(r)

    keys = [(r["idx"], r["rollout"]) for r in recs]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        sys.exit(f"ABORT: duplicate (idx, rollout) keys, halves overlap: {sorted(dupes)[:10]}")

    recs.sort(key=lambda r: (r["idx"], r["rollout"]))
    missing = sorted(set(range(N_EXPECTED)) - {r["idx"] for r in recs})
    if missing:
        msg = f"{len(missing)} problems missing (first: {missing[:5]})"
        if not args.force:
            sys.exit(f"ABORT: {msg}. Resume may still be running; use --force to write anyway.")
        print(f"  WARNING: {msg}")

    with open(MERGED, "wb") as fh:
        with zstd.ZstdCompressor(level=3).stream_writer(fh) as w:
            for r in recs:
                w.write((json.dumps(r) + "\n").encode())

    n_ok = sum(r["correct"] for r in recs)
    trunc = sum(r["num_steps"] >= r["sampling"]["max_new_tokens"] or r["pred_answer"] is None
                for r in recs)
    print(f"\nwrote {MERGED}: {len(recs)} records, {os.path.getsize(MERGED)/1e6:.1f} MB")
    print(f"  accuracy   {n_ok/len(recs):.4f}")
    print(f"  truncated  {trunc} ({trunc/len(recs):.1%}) -- exclude these before any "
          f"entropy-vs-correctness claim")


if __name__ == "__main__":
    main()
