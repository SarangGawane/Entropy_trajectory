"""Read entropy-acc-traces JSONL.zst files.

Each line is one full generation trace. Records carry the generated text/tokens
plus a `per_step` block of per-token signals: entropy, varentropy, top1_prob,
top2_prob, margin, self_certainty, p_tail, top20_token_ids, top20_logprobs.

Reader adapted from the dataset card.
"""

import json
import zstandard as zstd


def read_jsonl_zst(path):
    """Yield one parsed JSON record per line from a zstd-compressed JSONL file."""
    with open(path, "rb") as f:
        r = zstd.ZstdDecompressor().stream_reader(f, read_across_frames=True)
        buf = b""
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while (nl := buf.find(b"\n")) >= 0:
                line, buf = buf[:nl], buf[nl + 1:]
                if line.strip():
                    yield json.loads(line)
        if buf.strip():
            yield json.loads(buf)


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    for i, rec in enumerate(read_jsonl_zst(path)):
        if i == 0:
            print("top-level keys:", list(rec.keys()))
            if "per_step" in rec:
                ps = rec["per_step"]
                keys = list(ps.keys()) if isinstance(ps, dict) else "(list)"
                print("per_step keys:", keys)
        if i >= 2:
            break
    print("OK — file reads cleanly")
