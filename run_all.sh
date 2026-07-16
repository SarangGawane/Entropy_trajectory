#!/usr/bin/env bash
# Full trace generation: AIME24 then MATH-500 on Qwen3-8B bf16, single 4090.
# Sequential by necessity -- one GPU, and each config is sized to fill its VRAM.
set -u
cd /workspace/Ent_Traj
PY=.venv/bin/python
mkdir -p out

# batch/context sized so weights(16.4G) + KV stays under ~22G of the 24.5G card:
#   aime24 : 2 seqs x 16384 tok x 0.1406 MB/tok = 4.6G  -> ~21G
#   math500: 4 seqs x  8192 tok x 0.1406 MB/tok = 4.6G  -> ~21G
echo "=== AIME24 $(date -u +%FT%TZ) ==="
$PY generate_traces.py --bench aime24 \
    --batch-size 2 --max-new-tokens 16384 \
    --out out/aime24.jsonl.zst 2>&1 | grep -v "it/s\]$"

echo "=== MATH500 $(date -u +%FT%TZ) ==="
$PY generate_traces.py --bench math500 \
    --batch-size 4 --max-new-tokens 8192 \
    --out out/math500.jsonl.zst 2>&1 | grep -v "it/s\]$"

echo "=== ALL DONE $(date -u +%FT%TZ) ==="
