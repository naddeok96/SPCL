#!/usr/bin/env bash
# Fix and merge transition histories from run_evolution_parallel.sh
# Usage: ./fix_parallel_history.sh [HISTORY_DIR] [OUTPUT_PT]

set -euo pipefail

HIST_DIR="${1:-vec_evo_results_parallel/history}"
OUTPUT="${2:-vec_evo_results_parallel/fixed_history.pt}"

python reorganize_parallel_history.py --history_dir "$HIST_DIR" --output "$OUTPUT"

echo "Fixed history written to $OUTPUT"

