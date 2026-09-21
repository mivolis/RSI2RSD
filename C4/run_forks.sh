#!/bin/bash
# Run only after the paired pilot has finished. One prespecified fork per GPU.
set -euo pipefail
cd /workspace/c4
test -f PILOT_WORKERS_FINISHED
test ! -f STOP
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/workspace/c4/mplcache
horizon=${1:-20}
case "$horizon" in 20|94|1410) ;; *) exit 2 ;; esac
mkdir -p results/forks
CUDA_VISIBLE_DEVICES=0 python -u forks.py --parent results/gamma1 --data data/CIFAR-100-C --after-block 5 --horizon "$horizon" --output "results/forks/after5_h${horizon}" > "logs/fork_after5_h${horizon}.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python -u forks.py --parent results/gamma1 --data data/CIFAR-100-C --after-block 15 --horizon "$horizon" --output "results/forks/after15_h${horizon}" > "logs/fork_after15_h${horizon}.log" 2>&1 &
p1=$!
printf '%s\n%s\n' "$p0" "$p1" > fork_worker_pids.txt
e0=0;e1=0
wait "$p0" || e0=$?
wait "$p1" || e1=$?
printf '{"after5_exit":%s,"after15_exit":%s,"horizon":%s}\n' "$e0" "$e1" "$horizon" > "fork_h${horizon}_exit.json"
test "$e0" -eq 0 && test "$e1" -eq 0
python verify_forks.py "results/forks/after5_h${horizon}" > "logs/verify_after5_h${horizon}.log" 2>&1
python verify_forks.py "results/forks/after15_h${horizon}" > "logs/verify_after15_h${horizon}.log" 2>&1
date -u > "FORK_H${horizon}_COMPLETE"
