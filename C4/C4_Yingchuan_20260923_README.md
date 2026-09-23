# C4 Yingchuan sequential sweep (2026-09-23)

This directory holds Yingchuan's assigned 40 exploratory cells from `Split (1).pdf`. Run lambda `0.4` first, then `0.7`; at each lambda, beta goes `0.0005` -> `0.001` -> `0.0015` -> `0.002`. Each pair has seeds `202, 303, 404, 505, 606`. Finish and audit all five seeds, then issue one English PDF **before** starting the next beta. Do not confuse `0.0005` with `0.005`. Checkpoints 1 and 2 already have Sapelo arrays **48468709** and **48481042**; do not submit them again. Their [first](C4_Yingchuan_Pair01_README.md) and [second](C4_Yingchuan_Pair02_README.md) report packages are available here.

Checkpoint 3 (lambda=0.4, beta=0.0015; Sapelo array 48482278) is audited and delivered: [English PDF](C4_Yingchuan_Pair03_l04_b00015_EN.pdf), [compact evidence and checksums](C4_Yingchuan_Pair03_README.md). All five seeds completed; third-return CE rose in 2/5 seeds while mean classification error fell. The closed Jacobian remains unidentified. Checkpoint 4 (lambda=0.4, beta=0.002; Sapelo array 48487700) is audited and delivered: [English PDF](C4_Yingchuan_Pair04_l04_b0002_EN.pdf), [compact evidence and checksums](C4_Yingchuan_Pair04_README.md). All five seeds completed; third-return CE rose in 4/5 seeds while mean classification error fell. The closed Jacobian remains unidentified. The requested second-person parameter glance remains unconfirmed. No later pair is represented as completed.

The frozen C4 learner and fork code are in `code/`. Only the `--gamma` CLI choices in `c4.py` and `dynamics_v1/extend_trajectory.py` were extended to accept `0.4` and `0.7`. `code/run_pair_cell.py` orchestrates one seed: one-cycle parent, block-5/15 H94/H1410 forks, H20 dynamics, then the remaining 30 trajectory blocks. Every component has a separate status and log; the worker's exit code alone is not a scientific completion certificate.

## Sapelo setup

The existing C4 Python environment, verified CIFAR-100-C data, source-model checkpoint and CoTTA vendor source live at `/scratch/yz54720/RSI2RSD/C4/`. Copy `code/` into a **new** workspace. The script below is site-specific; adjust its absolute paths and Slurm resource choices before use. Sapelo's operator/login host is for `sbatch`; `xfer.gacrc.uga.edu` is for transfer only.

```bash
workspace=/scratch/yz54720/RSI2RSD/C4/sweep_yingchuan_20260923
mkdir -p "$workspace/logs" "$workspace/results"
cp -a C4/Yingchuan_20260923/code/. "$workspace/"
cd "$workspace"
for item in data checkpoints vendor; do ln -s "../$item" "$item"; done
module load Python/3.11.3-GCCcore-12.3.0
source ../.venv/bin/activate
python test_c4.py
python test_forks.py
python dynamics_v1/test_dynamics.py
python dynamics_v1/test_analysis.py
```

For a new checkpoint, create exactly one `pair_XX.json` from the assigned matrix. Compare its lambda, beta and five seeds with the PDF; have a second person glance at the launch values as requested there. Check existing job receipts to prevent duplicates, run `sbatch --test-only` on eligible resource options, and keep C4 concurrency at or below three GPUs. The **first** pair used:

```bash
sbatch --array=0-4%3 \
  --export=ALL,C4_PAIR_CONFIG="$workspace/pair_01.json" \
  "$workspace/full_pair.sbatch"
```

The included `full_pair.sbatch` requests `zhai_p`, account `xz2lab`, one L40S per task, four CPUs, 24 GB and five hours. These are first-pair Sapelo settings, not universal requirements. Array indices 0-4 map to seeds 202/303/404/505/606. Record the job ID, config/script hashes, logs and output path in a `pair_XX_submission.json`. Never prequeue later beta values.

## Evidence and report gate

After all five tasks have **terminal `sacct` states**, reconcile each worker status, parent/extension block count, fork verification, H94 shared prefix, H20 finite-response count and manifest verification. Failed attempts and missing descendants remain explicit. From the remote workspace, for the first pair:

```bash
python prepare_pair_evidence.py l04_b00005 --array 48468709
```

Transfer only `compact/l04_b00005/` through xfer to `C4/Yingchuan_20260923/Evidence/l04_b00005/` or the bound project's matching experiment folder. Bulk checkpoints, logits and raw step tapes stay on Sapelo. With ReportLab available, run `build_pair_report.py l04_b00005` from its parent directory. It writes `Reports/l04_b00005/`: a landscape-A4 English PDF, Markdown summary and complete block/cycle/return/fork/dynamics CSVs. Render and inspect the PDF before starting the next pair.

Each PDF reports C4-1 through C4-7: local proxy/labelled signs; student/teacher and pseudo-supervision trajectories; same-domain returns; per-seed cost/outcome summary; immediate and H94/H1409/H1410 branch contrasts; `x_t`, empirical `F`, and perturbation/identifiability checks; and a bounded parameter-map reading. Unavailable quantities and numerical failures are marked, not imputed. A finite response or endpoint contrast is not an identified Jacobian, phase boundary or persistent-deterioration result. The confirmation panel remains untouched.
