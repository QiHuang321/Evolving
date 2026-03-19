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

# ── Accuracy (single-run) ─────────────────────────────────────────────────────
run_eval starting_scripts     # expected: 0.3828
run_eval manual_instruct      # expected: 0.5391
run_eval manual_thinking      # expected: 0.5078
run_eval evolved_instruct     # expected: 0.6094
run_eval evolved_thinking     # expected: 0.5547
run_eval best_accuracy        # expected: 0.7969
run_eval best_speed           # expected: 0.5781 (batched)
run_eval best_overall         # expected: 0.7813

# ── Fold-wise stability check ─────────────────────────────────────────────────
echo "===== best_accuracy --cv 4 ====="
python evaluate.py best_accuracy --cv 4
echo

echo "===== best_overall --cv 4 ====="
python evaluate.py best_overall --cv 4
echo

# ── Speed benchmark (5 repeated runs) ─────────────────────────────────────────
echo "===== best_speed timing (5 runs) ====="
for i in 1 2 3 4 5; do
  echo "--- run $i ---"
  python evaluate.py best_speed
done
echo
