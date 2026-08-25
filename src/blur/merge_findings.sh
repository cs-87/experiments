#!/usr/bin/env bash
# Merge the per-shard findings into the canonical files. Idempotent: run it any time,
# mid-sweep included, to see everything measured so far.
set -uo pipefail
cd /home/ubuntu/cs_exp/experiments
shards=$(ls src/blur/FINDINGS.r*.md src/blur/FINDINGS.vis.md src/blur/FINDINGS.fp.md 2>/dev/null)
if [ -z "$shards" ]; then
  # Without this the redirect below would truncate a good FINDINGS.md to nothing the
  # first time this runs somewhere the per-shard files are absent.
  echo "no per-shard findings to merge; leaving src/blur/FINDINGS.md alone" >&2
  exit 0
fi
{
  for R in 80 110 140 200; do
    [ -f "src/blur/FINDINGS.r$R.md" ] && cat "src/blur/FINDINGS.r$R.md"
  done
  [ -f src/blur/FINDINGS.vis.md ] && cat src/blur/FINDINGS.vis.md
  [ -f src/blur/FINDINGS.fp.md ]  && cat src/blur/FINDINGS.fp.md
} > src/blur/FINDINGS.md 2>/dev/null
cat src/blur/findings.r*.jsonl src/blur/findings.vis.jsonl src/blur/findings.fp.jsonl \
    2>/dev/null > src/blur/findings.jsonl
echo "merged: $(grep -hc . src/blur/findings.jsonl 2>/dev/null || echo 0) rows -> src/blur/FINDINGS.md"
