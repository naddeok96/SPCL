#!/usr/bin/env bash
set -euo pipefail

# Activate the virtual environment
VENV_PATH="../envs/spcl"
echo "Activating venv at $VENV_PATH"
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

# Configuration and directories
CONFIG="config.yaml"
GPUS=(0 1)
RUNS_PER_GPU=1
POP_SIZE=256
GENERATIONS=16

SAVE_PATH="saved_model_checkpoints"
POP_DIR="populations"
EVAL_DIR="eval_parts"
HIST_DIR="history"
LOG_DIR="logs"
PID_FILE="pids.txt"

# Clean any old PID file and recreate directories
rm -f "$PID_FILE"
mkdir -p "$POP_DIR" "$EVAL_DIR" "$HIST_DIR" "$SAVE_PATH" "$LOG_DIR"

# 0) Reserve each GPU with a tiny placeholder and record its PID
for gpu in "${GPUS[@]}"; do
  echo "Reserving GPU $gpu with placeholder" | tee -a "$LOG_DIR/placeholder.log"

  nohup bash -c "\
    CUDA_VISIBLE_DEVICES=$gpu python -c '\
import torch, time
torch.cuda.init()
_ = torch.zeros((1,), device=\"cuda\")
time.sleep(86400)
'" \
    > "$LOG_DIR/placeholder_gpu${gpu}.log" 2>&1 &

  # capture the placeholder’s PID
  echo $! >> "$PID_FILE"
done

# 1) Initialize Generation 0 population
python init_population.py --config "$CONFIG" --output "$POP_DIR/pop_gen_0.npz" 2>&1 | tee "$LOG_DIR/init_population.log"

# 2) Evolutionary loop
for (( gen=0; gen<GENERATIONS; gen++ )); do
  echo "=== Generation $gen ===" | tee -a "$LOG_DIR/run_evolution.log"
  POP_FILE="$POP_DIR/pop_gen_${gen}.npz"

  # Compute how to split the population across jobs
  TOTAL_RUNS=$(( ${#GPUS[@]} * RUNS_PER_GPU ))
  BASE=$(( POP_SIZE / TOTAL_RUNS ))
  REM=$(( POP_SIZE % TOTAL_RUNS ))
  run_idx=0

  # Remove any leftover eval parts for this generation
  rm -f "$EVAL_DIR"/eval_gen${gen}_part*.npz

  # 2a) Launch parallel evaluation jobs
  for gpu in "${GPUS[@]}"; do
    for _ in $(seq 1 $RUNS_PER_GPU); do
      start=$(( run_idx * BASE ))
      cnt=$BASE
      if [ "$run_idx" -lt "$REM" ]; then
        cnt=$(( cnt + 1 ))
      fi
      OUT="$EVAL_DIR/eval_gen${gen}_part${run_idx}.npz"
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
  python merge_history.py --history_dir "$EVAL_DIR" --output "$HIST_DIR/history_gen${gen}.npz" 2>&1 | tee -a "$LOG_DIR/merge_history_gen${gen}.log"

  # 4) Evolve the next generation population
  NEXT_POP="$POP_DIR/pop_gen_$((gen+1)).npz"
  python evolve_population.py --config "$CONFIG" --pop_file "$POP_FILE" --eval_dir "$EVAL_DIR" --gen "$gen" --output_population "$NEXT_POP" --history_dir "$HIST_DIR" 2>&1 | tee -a "$LOG_DIR/evolve.log"
done

# 5) Final merge of all per-generation history segments into one dataset
FINAL_DS="$SAVE_PATH/evolutionary_dataset.npz"
python merge_history.py --history_dir "$EVAL_DIR" --output "$FINAL_DS" 2>&1 | tee -a "$LOG_DIR/merge_history.log"

# 6) Cleanup: kill GPU placeholder processes
echo "Cleaning up GPU placeholders…" | tee -a "$LOG_DIR/run_evolution.log"
pkill -f "placeholder_gpu" || true

echo "✅ Done. All logs in $LOG_DIR, PIDs in $PID_FILE"