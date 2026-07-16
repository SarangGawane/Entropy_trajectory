"""Download the entropy-acc-traces dataset into this directory.

The dataset is gated (manual approval). Before running:
  1. Request access at https://huggingface.co/datasets/Shichengf/entropy-acc-traces
  2. Wait for the author to approve.
  3. Authenticate: `hf auth login`  (or export HF_TOKEN=hf_...)

Usage:
  python download.py              # all files (~3.9 GB)
  python download.py aime24       # only files matching a substring
"""

import sys
from huggingface_hub import snapshot_download

REPO = "Shichengf/entropy-acc-traces"
HERE = __file__.rsplit("/", 1)[0]

FILES = [
    "qwen3-8b-instruct-sft/aime24.jsonl.zst",
    "qwen3-8b-instruct-sft/aime24_stage2_mt16384.jsonl.zst",
    "qwen3-8b-instruct-sft/math500.jsonl.zst",
    "qwen3-8b-base-sft/aime24.jsonl.zst",
    "qwen3-8b-base-sft/math500.jsonl.zst",
]


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    wanted = [f for f in FILES if not filt or filt in f]
    if not wanted:
        sys.exit(f"No files match {filt!r}. Available:\n  " + "\n  ".join(FILES))

    print(f"Downloading {len(wanted)} file(s) to {HERE}")
    path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=HERE,
        allow_patterns=wanted,
    )
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
