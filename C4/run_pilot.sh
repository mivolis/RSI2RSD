#!/bin/bash
set -euo pipefail
cd /workspace/c4
test -f PREFLIGHT_COMPLETE
test ! -f STOP
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/workspace/c4/mplcache
CUDA_VISIBLE_DEVICES=0 python -u c4.py --data data/CIFAR-100-C --gamma 0 --output results/gamma0 > logs/gamma0.log 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 python -u c4.py --data data/CIFAR-100-C --gamma 1 --output results/gamma1 > logs/gamma1.log 2>&1 &
pid1=$!
printf '%s\n%s\n' "$pid0" "$pid1" > worker_pids.txt
exit0=0;exit1=0
wait "$pid0" || exit0=$?
wait "$pid1" || exit1=$?
printf '{"gamma0_exit":%s,"gamma1_exit":%s}\n' "$exit0" "$exit1" > pilot_exit.json
python report.py results --verify > logs/final_verification.log 2>&1
if [ "$exit0" -eq 0 ] && [ "$exit1" -eq 0 ]; then
  date -u > PILOT_WORKERS_FINISHED
fi
