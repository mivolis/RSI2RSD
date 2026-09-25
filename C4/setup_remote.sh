#!/bin/bash
set -euo pipefail
cd /workspace/c4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/workspace/c4/mplcache
mkdir -p logs results preflight
if ! command -v git >/dev/null || ! command -v rsync >/dev/null; then
  apt-get update > logs/apt.log 2>&1
  apt-get install -y git rsync >> logs/apt.log 2>&1
fi
if [ ! -d vendor/cotta/.git ]; then
  mkdir -p vendor
  git clone https://github.com/qinenergy/cotta.git vendor/cotta > logs/cotta.log 2>&1
  git -C vendor/cotta checkout c212a204b32be4005092e4323105a24a29ad2952 >> logs/cotta.log 2>&1
fi
python -m pip install -r requirements.txt > logs/install.log 2>&1
python test_c4.py > logs/contracts.log 2>&1
if [ -f data/CIFAR-100-C/input_manifest.json ]; then
  python - <<'PYDATA' > logs/data.log 2>&1
import json
from pathlib import Path
from c4 import filehash
from prepare_data import MD5
root=Path('data/CIFAR-100-C');m=json.loads((root/'input_manifest.json').read_text())
assert m['source']['md5']==MD5
for name,sha in m['files'].items():assert filehash(root/name)==sha,name
print('Verified all severity-5 extraction hashes; source archive',MD5)
PYDATA
  cp data/CIFAR-100-C/input_manifest.json input_data_provenance.json
else
  python prepare_data.py data > logs/data.log 2>&1
  cp data/data_provenance.json input_data_provenance.json
fi
python - <<'PY' > logs/source.log 2>&1
from robustbench.utils import load_model
model=load_model(model_name='Hendrycks2020AugMix_ResNeXt', dataset='cifar100', threat_model='corruptions',model_dir='checkpoints').eval()
print('Source loaded',sum(p.numel() for p in model.parameters()))
PY
date -u > SETUP_COMPLETE
