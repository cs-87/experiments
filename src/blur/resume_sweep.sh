#!/usr/bin/env bash
# Restart the sweep after any stop. Safe to run repeatedly: --resume skips every cell
# already in the per-shard findings jsonl, and marked videos/transcodes are cached on
# disk, so nothing is re-encoded and nothing is re-measured.
#
#   tmux new -s blursweep -d "bash $0"
set -uo pipefail
cd /home/ubuntu/cs_exp/experiments
PY=myenv/bin/python
LOGS="/tmp/claude-1000/-home-ubuntu-cs-exp-experiments/01498f06-f779-447e-80d0-bbf5b05c5926/scratchpad/logs"
SP="/tmp/claude-1000/-home-ubuntu-cs-exp-experiments/01498f06-f779-447e-80d0-bbf5b05c5926/scratchpad"
mkdir -p "$LOGS"
export BLUR_DCT_WORKERS=2 PYTHONUNBUFFERED=1

TRS="1,3,5,10,30"
ALL="clean,crf23,blur_only,moire_only,mild,moderate,severe"

shard () {
  local R=$1
  {
    echo "=== shard R=$R resumed $(date -Is) ==="
    $PY -m src.blur.eval_harness sweep --radius "$R" --tr "$TRS" \
        --conditions "$ALL" --lightglue --tag "r$R" --resume --quiet
    echo "=== shard R=$R finished $(date -Is) ==="
  } >> "$LOGS/sweep_r$R.log" 2>&1
}

echo "resume started $(date -Is)"
for R in 80 110 140 200; do shard "$R" & done
wait
echo "all sweep shards done $(date -Is)"

[ -f src/blur/FINDINGS.vis.md ] || $PY -m src.blur.eval_harness visibility \
    --radius 80,110,140,200 --tr 1,30 --dump outputs/harness/dump --tag vis --quiet \
    >> "$LOGS/visibility.log" 2>&1
echo "visibility done $(date -Is)"

[ -f src/blur/FINDINGS.fp.md ] || $PY -m src.blur.eval_harness fp \
    --radius 110,140 --tr 1,30 --tag fp --quiet >> "$LOGS/fp.log" 2>&1
echo "fp done $(date -Is)"

bash "$SP/merge_findings.sh"
echo "ALL DONE $(date -Is)"
