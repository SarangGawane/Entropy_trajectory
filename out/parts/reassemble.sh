#!/usr/bin/env bash
# out/math500.jsonl.zst is 262 MB, past GitHub's 100 MB per-file limit, so it is
# stored split. This puts it back together and verifies it.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "reassembling math500.jsonl.zst from $(ls parts/*.part-* | wc -l) parts..."
cat parts/math500.jsonl.zst.part-* > math500.jsonl.zst
sha256sum -c parts/math500.jsonl.zst.sha256
echo "OK: $(du -h math500.jsonl.zst | cut -f1) reassembled and verified"
