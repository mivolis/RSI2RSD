#!/bin/bash -l
#SBATCH --job-name=c4-feedback-forks
#SBATCH --partition=zhai_p
#SBATCH --account=xz2lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/yz54720/RSI2RSD/C4
#SBATCH --output=/scratch/yz54720/RSI2RSD/C4/logs/slurm-%j.out
#SBATCH --error=/scratch/yz54720/RSI2RSD/C4/logs/slurm-%j.err
set -euo pipefail
trap 'rc=$?; printf "{\"exit_code\":%s,\"slurm_job_id\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" > "logs/job_${SLURM_JOB_ID}_exit.json"' EXIT
module load Python/3.11.3-GCCcore-12.3.0
if [ ! -x .venv/bin/python ]; then python -m venv .venv; fi
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$PWD/mplcache"
export PIP_CACHE_DIR="$PWD/pipcache"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
python -m pip install --upgrade pip==25.0.1 --index-url https://pypi.org/simple > "logs/pip_bootstrap_${SLURM_JOB_ID}.log" 2>&1
python -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124 > "logs/pytorch_${SLURM_JOB_ID}.log" 2>&1
python -m pip install -r requirements.txt pillow==10.2.0 > "logs/deps_${SLURM_JOB_ID}.log" 2>&1
python -m pip freeze > "logs/pip_${SLURM_JOB_ID}.txt"
nvidia-smi > "logs/gpu_${SLURM_JOB_ID}.txt"
for attempt in $(seq 1 60); do
  test ! -f INPUTS_VERIFIED || break
  sleep 5
done
test -f INPUTS_VERIFIED
python test_c4.py > "logs/tests_natural_${SLURM_JOB_ID}.log" 2>&1
python test_forks.py > "logs/tests_forks_${SLURM_JOB_ID}.log" 2>&1
python -u run_sapelo_forks.py
