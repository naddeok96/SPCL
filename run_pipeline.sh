#!/usr/bin/env bash
set -euo pipefail

# ─── Toggle which steps to run ───────────────────────────────────────────────
RUN_DATASET=false   # set to false to skip generate_dataset.py
RUN_OFFPOLICY=true    # set to false to skip off_policy_train.py
RUN_ONPOLICY=true     # set to false to skip on_policy_train.py
# ────────────────────────────────────────────────────────────────────────────

CONFIG="config.yaml"
VENV_PATH="../envs/spcl"

echo "Activating venv at $VENV_PATH"
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

echo "Using config: $CONFIG"
echo

if [ "$RUN_DATASET" = true ]; then
  echo "=== Step 1/3: Generating EA dataset ==="
  python generate_dataset.py --config "$CONFIG"
  echo
fi

if [ "$RUN_OFFPOLICY" = true ]; then
  echo "=== Step 2/3: Off‑policy training ==="
  python off_policy_train.py --config "$CONFIG"
  echo
fi

if [ "$RUN_ONPOLICY" = true ]; then
  echo "=== Step 3/3: On‑policy training ==="
  python on_policy_train.py --config "$CONFIG"
  echo
fi

echo "Pipeline complete."
