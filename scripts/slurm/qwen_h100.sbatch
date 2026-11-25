#!/bin/bash
#SBATCH --job-name=prestige-qwen
#SBATCH --partition=h100hour
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH --gres=gpu:1
#SBATCH --output=/local/scratch/%u/prestige-mad/logs/%x-%j.out
# The `qwen` stage of scripts/run_prestige_v2.sh on one H100. Same python command; differences:
#   N=600, WARMUP=50 passed explicitly (the script's own defaults resolve to 120/20),
#   servers started here (the cluster forbids setting CUDA_VISIBLE_DEVICES), context 16384.
# Resumable: resubmit to continue from the same store.
set -eo pipefail
. /local/scratch/$USER/prestige-mad/code/scripts/slurm/env_h100.sh
. "$PROJ/venv/bin/activate"
export HF_HUB_OFFLINE=1
N=600; WARMUP=50; ROUNDS=3
EXPERT=Qwen/Qwen2.5-7B-Instruct; WEAK=Qwen/Qwen2.5-3B-Instruct
V="$PROJ/logs/vllm-$SLURM_JOB_ID"; mkdir -p "$V" runs
# design expert_open hard-codes ports 8001/8002; never talk to someone else's server
python - <<'PY'
import socket
for p in (8001, 8002):
    with socket.socket() as s: s.bind(("127.0.0.1", p))
PY
PIDS=(); trap 'kill "${PIDS[@]}" 2>/dev/null || true; wait 2>/dev/null || true' EXIT
serve() {   # model port fraction
  vllm serve "$1" --host 127.0.0.1 --port "$2" --max-model-len 16384 --gpu-memory-utilization "$3" > "$V/$2.log" 2>&1 &
  local pid=$!; PIDS+=("$pid")
  for _ in $(seq 240); do
    curl -fsS "http://127.0.0.1:$2/v1/models" 2>/dev/null | grep -q "$1" && { echo "ready $1 $(date +%T)"; return; }
    kill -0 "$pid" 2>/dev/null || { echo "server $1 died"; tail -30 "$V/$2.log"; exit 1; }
    sleep 3
  done
  echo "server $1 timed out"; exit 1
}
date; nvidia-smi -L
serve "$EXPERT" 8001 0.45
serve "$WEAK" 8002 0.35
python prestige_mad_v2.py --design expert_open --expert-model "$EXPERT" --weak-model "$WEAK" \
  --problems logs/math_extra.jsonl --level 4,5 --n $N --warmup $WARMUP --rounds $ROUNDS \
  --workers 24 --stagger 0 --max-retries 8 --max-tokens 2048 --policies prestige,all_speak,none \
  --log "$PROJ/logs/calls.jsonl" --store runs/prestige_qwen_math45_n$N.jsonl
date
