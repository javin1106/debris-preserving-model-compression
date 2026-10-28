#!/usr/bin/env bash
# Live view of training progress: one line per finished round for every run in a job file.
# Uses polling, because inotify misses writes to files under /mnt/c in WSL.
# Usage (inside WSL, repo root):  bash scripts/watch_runs.sh JOBFILE
cd "$(dirname "$0")/.."
logs=()
while read -r rid _; do
    [ -z "$rid" ] || [[ "$rid" == \#* ]] && continue
    logs+=("runs/$rid/train.log")
done < "$1"
tail -n 2 --use-polling -s 5 -F "${logs[@]}" 2>/dev/null | grep --line-buffered -E "==>|epoch [0-9]+ \||FINAL|DONE|Error"
