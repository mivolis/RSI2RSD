#!/bin/bash
set -euo pipefail
cd /workspace/c4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/workspace/c4/mplcache
CUDA_VISIBLE_DEVICES=0 python -u c4.py --data data/CIFAR-100-C --gamma 0 --cycles 1 --domains 2 --max-batches 2 --output preflight/gamma0 > logs/preflight_gamma0.log 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 python -u c4.py --data data/CIFAR-100-C --gamma 1 --cycles 1 --domains 2 --max-batches 2 --output preflight/gamma1 > logs/preflight_gamma1.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
python report.py preflight --verify > logs/preflight_verification.log 2>&1
date -u > PREFLIGHT_COMPLETE
