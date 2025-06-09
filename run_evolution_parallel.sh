#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="../envs/spcl"
echo "Activating venv at $VENV_PATH"
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

CONFIG="config.yaml"
GPUS=(0 1 2 3 4)
RUNS_PER_GPU=1
POP_SIZE=12
GENERATIONS=2
NUM_MODELS=4

SAVE_PATH="evo_results_parallel/saved_model_checkpoints"
POP_DIR="evo_results_parallel/populations"
EVAL_DIR="evo_results_parallel/eval_parts"
HIST_DIR="evo_results_parallel/history"
LOG_DIR="evo_results_parallel/logs"
PID_FILE="evo_results_parallel/pids.txt"

rm -f "$PID_FILE"
mkdir -p "$POP_DIR" "$EVAL_DIR" "$HIST_DIR" "$SAVE_PATH" "$LOG_DIR"

python init_population.py --config "$CONFIG" --output "$POP_DIR/pop_gen_0.pt" 2>&1 | tee "$LOG_DIR/init_population.log"

for (( gen=0; gen<GENERATIONS; gen++ )); do
  echo "=== Generation $gen ===" | tee -a "$LOG_DIR/run_evolution_parallel.log"
  POP_FILE="$POP_DIR/pop_gen_${gen}.pt"

  TOTAL_RUNS=$(( ${#GPUS[@]} * RUNS_PER_GPU ))
  BASE=$(( POP_SIZE / TOTAL_RUNS ))
  REM=$(( POP_SIZE % TOTAL_RUNS ))
  run_idx=0

  rm -f "$EVAL_DIR"/eval_gen${gen}_part*.pt

  for gpu in "${GPUS[@]}"; do
    for _ in $(seq 1 $RUNS_PER_GPU); do
      start=$(( run_idx * BASE ))
      cnt=$BASE
      if [ "$run_idx" -lt "$REM" ]; then
        cnt=$(( cnt + 1 ))
      fi
      OUT="$EVAL_DIR/eval_gen${gen}_part${run_idx}.pt"
      LOGF="$LOG_DIR/eval_gen${gen}_part${run_idx}.log"

      echo "Launching eval gen $gen part $run_idx → GPU $gpu (idx $start cnt $cnt)" | tee -a "$LOG_DIR/run_evolution_parallel.log"
      CUDA_VISIBLE_DEVICES=$gpu python eval_population.py \
        --config "$CONFIG" --pop_file "$POP_FILE" \
        --start_idx "$start" --num_candidates "$cnt" \
        --out_file "$OUT" --num_models "$NUM_MODELS" --model_type mlp \
        > "$LOGF" 2>&1 &
      echo $! >> "$PID_FILE"
      run_idx=$(( run_idx + 1 ))
    done
  done

  echo "Waiting for all $TOTAL_RUNS eval jobs…" | tee -a "$LOG_DIR/run_evolution_parallel.log"
  wait
  echo "All evals for gen $gen done." | tee -a "$LOG_DIR/run_evolution_parallel.log"

  python merge_history.py --history_dir "$EVAL_DIR" --output "$HIST_DIR/history_gen${gen}.pt" 2>&1 | tee -a "$LOG_DIR/merge_history_gen${gen}.log"

  NEXT_POP="$POP_DIR/pop_gen_$((gen+1)).pt"
  python evolve_population.py --config "$CONFIG" --pop_file "$POP_FILE" --eval_dir "$EVAL_DIR" --gen "$gen" --output_population "$NEXT_POP" --history_dir "$HIST_DIR" 2>&1 | tee -a "$LOG_DIR/evolve.log"
done

FINAL_DS="$SAVE_PATH/evolutionary_dataset.pt"
python merge_history.py --history_dir "$EVAL_DIR" --output "$FINAL_DS" 2>&1 | tee -a "$LOG_DIR/merge_history.log"

rm -f "$PID_FILE"

echo "✅ Parallel evolution run complete. Logs in $LOG_DIR"
