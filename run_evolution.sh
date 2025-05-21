#!/usr/bin/env bash
set -euo pipefail

# Activate the virtual environment
VENV_PATH="../envs/spcl"
echo "Activating venv at $VENV_PATH"
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

# Configuration and directories
CONFIG="config.yaml"
GPUS=(0 1 2 3 4)
RUNS_PER_GPU=1
POP_SIZE=12
GENERATIONS=2

SAVE_PATH="evo_results/saved_model_checkpoints"
POP_DIR="evo_results/populations"
EVAL_DIR="evo_results/eval_parts"
HIST_DIR="evo_results/history"
LOG_DIR="evo_results/logs"
PID_FILE="evo_results/pids.txt"

# Clean any old PID file and recreate directories
rm -f "$PID_FILE"
mkdir -p "$POP_DIR" "$EVAL_DIR" "$HIST_DIR" "$SAVE_PATH" "$LOG_DIR"

# 1) Initialize Generation 0 population
python init_population.py --config "$CONFIG" --output "$POP_DIR/pop_gen_0.pt" 2>&1 | tee "$LOG_DIR/init_population.log"

# 2) Evolutionary loop
for (( gen=0; gen<GENERATIONS; gen++ )); do
  echo "=== Generation $gen ===" | tee -a "$LOG_DIR/run_evolution.log"
  POP_FILE="$POP_DIR/pop_gen_${gen}.pt"

  # Compute how to split the population across jobs
  TOTAL_RUNS=$(( ${#GPUS[@]} * RUNS_PER_GPU ))
  BASE=$(( POP_SIZE / TOTAL_RUNS ))
  REM=$(( POP_SIZE % TOTAL_RUNS ))
  run_idx=0

  # Remove any leftover eval parts for this generation
  rm -f "$EVAL_DIR"/eval_gen${gen}_part*.pt

  # 2a) Launch parallel evaluation jobs
  for gpu in "${GPUS[@]}"; do
    for _ in $(seq 1 $RUNS_PER_GPU); do
      start=$(( run_idx * BASE ))
      cnt=$BASE
      if [ "$run_idx" -lt "$REM" ]; then
        cnt=$(( cnt + 1 ))
      fi
      OUT="$EVAL_DIR/eval_gen${gen}_part${run_idx}.pt"
      LOGF="$LOG_DIR/eval_gen${gen}_part${run_idx}.log"

      echo "Launching eval gen $gen part $run_idx → GPU $gpu (idx $start cnt $cnt)" | tee -a "$LOG_DIR/run_evolution.log"
      CUDA_VISIBLE_DEVICES=$gpu python eval_population.py --config "$CONFIG" --pop_file "$POP_FILE" --start_idx "$start" --num_candidates "$cnt" --out_file "$OUT" > "$LOGF" 2>&1 &
      echo $! >> "$PID_FILE"
      run_idx=$(( run_idx + 1 ))
    done
  done

  # Wait for all eval jobs to finish
  echo "Waiting for all $TOTAL_RUNS eval jobs…" | tee -a "$LOG_DIR/run_evolution.log"
  wait
  echo "All evals for gen $gen done." | tee -a "$LOG_DIR/run_evolution.log"

  # 3) Merge eval parts into a per-generation history segment
  python merge_history.py --history_dir "$EVAL_DIR" --output "$HIST_DIR/history_gen${gen}.pt" 2>&1 | tee -a "$LOG_DIR/merge_history_gen${gen}.log"

  # 4) Evolve the next generation population
  NEXT_POP="$POP_DIR/pop_gen_$((gen+1)).pt"
  python evolve_population.py --config "$CONFIG" --pop_file "$POP_FILE" --eval_dir "$EVAL_DIR" --gen "$gen" --output_population "$NEXT_POP" --history_dir "$HIST_DIR" 2>&1 | tee -a "$LOG_DIR/evolve.log"
done

# 5) Final merge of all eval parts into one dataset
FINAL_DS="$SAVE_PATH/evolutionary_dataset.pt"
python merge_history.py \
  --history_dir "$EVAL_DIR" \
  --output "$FINAL_DS" 2>&1 | tee -a "$LOG_DIR/merge_history.log"


rm -f "$PID_FILE"

echo "✅ Done. All logs in $LOG_DIR, PIDs in $PID_FILE"



