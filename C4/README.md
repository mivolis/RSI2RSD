# C4: exploratory self-referential continual adaptation

For the Yingchuan 40-cell sequential extension, see [the run and report guide](C4_Yingchuan_20260923_README.md) and [the code archive](C4_Yingchuan_Code_20260923.tar.gz). Checkpoint 1 is running; each five-seed parameter pair will receive a separate audited English PDF before the next beta.

For the September 2026 **18-cell gamma x seed rerun** (`gamma = 0/0.5/1`, `beta = .001/.01`, seeds `202/303/404`), follow the [fresh-clone setup and exact run commands](C4_20260922_README.md#re-run-the-gamma-x-seed-experiment-from-github). The `Run and verify` section below describes the earlier seed-101 meeting pilot and its `/workspace/c4` wrappers.


This is the executable implementation of the approved C4 experiment. It is
**not unmodified CoTTA**: all student parameters are trainable, BN buffers are
frozen, and confidence-filtered hard pseudo-labels supervise strong views.
There is no source-weight restoration or outcome-label acceptance gate.

Current measured outcomes: [RESULTS.md](RESULTS.md). The accompanying
`compact_results.tar.gz` contains the original pilot and four short-fork
summaries; `sapelo_compact_results.tar.gz` adds all six Sapelo forks through
H1410. [ARTIFACTS.md](ARTIFACTS.md) records bulk-data availability and checksums.
All scheduled GPU work is complete; release uploads are a separate delivery step.

## Frozen meeting-pilot design

- Public CIFAR-100-C, severity 5, all 15 corruptions in the order in `c4.py`.
- RobustBench `Hendrycks2020AugMix_ResNeXt` source; 6,900,132 parameters.
- Source/EMA mixing weight lambda (`--gamma`) = 0 or 1; teacher refresh
  beta = .001; SGD learning rate .001, momentum .9, confidence threshold .9.
- Seed 101, two cycles, 30 complete blocks per arm; 6,000 adaptation images
  per block, batch 64 including the final partial batch: 2,820 batches per arm.
- Split base-image identities before corruption: adaptation 6,000 / reserved
  gate 1,000 / exploratory audit 1,500 / untouched confirmation 1,500.
- Fixed reference Q: the smallest exploratory-audit ID per class crossed with
  all 15 corruptions, giving the same 1,500 image-corruption pairs at every audit.

Repeated corruptions share image identities. Domains, batches, and forks from
one parent are not independent replications. Confirmation outcomes remain unused.

## Run and verify

The GPU run used the PyTorch 2.5.1 / CUDA 12.4 runtime image and two independent
RTX 4090 GPUs. Pin additional packages with `requirements.txt` and CoTTA commit
`c212a204b32be4005092e4323105a24a29ad2952`. Actual package versions and checkpoint
hashes are saved in each run's environment record.

```sh
bash setup_remote.sh
python test_forks.py
bash preflight_remote.sh
# Inspect the preflight, time one full block, then:
bash run_pilot.sh
python report.py results --verify
bash run_forks.sh 20
# Inspect identity checks and timing before extending:
bash run_forks.sh 94
bash run_forks.sh 1410
```

The shell wrappers use `/workspace/c4`. Outside that layout, use the Python
entry points and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA.
`test_c4.py` contains nine contract tests; `test_forks.py` contains three,
including branch-state isolation and detection of altered retained evidence.
Tests on synthetic images are software checks, not scientific results.

The official archive MD5 is `11f0ed0f1191edbf9fa23466ae6021d3`. The loader also
supports byte-identical severity-5 slices produced by `extract_severity5.py`;
their manifest verifies every file against the MD5-verified source archive.

## Prespecified feedback forks

After blocks 5 and 15 of the lambda=1 first cycle, use the first nonempty
next-domain update. Report both candidates regardless of their local sign.
Save complete accept/reject states, including optimizer momentum and teacher.
The reject closed-loop branch generates a complete target **and mask** tape
before either accept continuation is run. The accept/replay continuation uses
that external tape; both open and closed learners continue training on paired
images and augmentation keys.

`forks.py` reports fixed-Q CE contrasts `A_closed`, `A_open`, and their difference
`I_feedback`. Reject/replay identity is explicitly checked for the first 20
steps; equality at longer horizons follows from the deterministic recurrence
and is labeled as such. `verify_forks.py` verifies hashes, reconstructs CE from
raw logits, checks schedule/tape identity, and recomputes the contrasts.

## Retained evidence

Each batch has a compressed per-sample archive and flushed JSONL: IDs, labels,
targets, confidence, mask, source/teacher probabilities, before/after student
logits, and paired view keys. Every block saves full audit predictions,
fixed-Q predictions, sampled local probes, and a resumable student/teacher/
optimizer/RNG checkpoint. `complete.json` is written last and hashes its files.
Interrupted attempts and failed preflight logs are retained separately.

`report.py` produces five natural-pilot figures; each fork has an additional
feedback-contrast figure. Large raw records/checkpoints are delivered as
checksummed release attachments, indexed alongside compact results under `C4/`,
rather than as oversized Git objects. Repository visibility is unchanged.

## Evidence boundaries

Proxy decrease does not establish true improvement. A worse student than source
does not establish cumulative deterioration. A closed/replay contrast concerns
the specified intervention, not a universal causal effect. This one-seed pilot
does not establish hypercompression, inheritance loss, a Jacobian, a phase
transition, benchmark superiority, or general RSI failure. Null and improving
trajectories are valid outcomes.
