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

run_eval manual_instruct
run_eval manual_thinking
run_eval evolved_instruct
run_eval evolved_thinking
run_eval best_accuracy
run_eval best_speed
run_eval best_overall
