#!/usr/bin/env bash
# kill_all.sh — kills all eval_population jobs launched by run_evolution.sh

set -euo pipefail

PID_FILE="pids.txt"
if [ ! -f "$PID_FILE" ]; then
  echo "⚠️  PID file not found: $PID_FILE"
  exit 1
fi

echo "Killing all background jobs listed in $PID_FILE..."
while read -r pid; do
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" && echo " → killed PID $pid"
  else
    echo " → PID $pid not running"
  fi
done < "$PID_FILE"

# cleanup
rm -f "$PID_FILE"
echo "All tracked jobs have been killed and $PID_FILE removed."
