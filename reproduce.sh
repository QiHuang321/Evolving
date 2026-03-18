#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

run_eval() {
  local module="$1"
  echo "===== $module ====="
  python evaluate.py "$module"
  echo
}

run_eval manual_instruct     # expected: 0.5391
run_eval manual_thinking     # expected: 0.5078
run_eval evolved_instruct    # expected: 0.6094
run_eval evolved_thinking    # expected: 0.5547
run_eval best_accuracy       # expected: 0.7969
run_eval best_speed          # expected: 0.5781 (batched)
run_eval best_overall        # expected: 0.7813
