#!/usr/bin/env bash
# Re-run AIME24 with a 32768-token cap to remove the truncation confound seen at
# 16384 (8/30 traces hit the cap and never emitted a boxed answer, which scored
# them "incorrect" and inflated the entropy gap between correct and incorrect).
#
# batch 1 is required: 32768 tok x 0.1406 MB/tok = 4.6G KV + 16.4G weights ~= 21G.
# batch 2 would need ~25.6G and OOM the 24.5G card.
# Qwen3-8B allows max_position_embeddings=40960, so 32k generation is in range.
set -u
cd /workspace/Ent_Traj

# Wait for the in-flight MATH-500 run to release the GPU.
while pgrep -f "generate_traces.py --bench math500" > /dev/null; do sleep 60; done
sleep 30

echo "=== AIME24 mt32768 $(date -u +%FT%TZ) ==="
.venv/bin/python generate_traces.py --bench aime24 \
    --batch-size 1 --max-new-tokens 32768 \
    --out out/aime24_mt32768.jsonl.zst 2>&1 | grep -v "it/s\]$"
echo "=== AIME32K DONE $(date -u +%FT%TZ) ==="
